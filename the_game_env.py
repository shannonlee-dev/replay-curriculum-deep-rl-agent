from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from curriculum import validate_distribution


class TheGameEnv(gym.Env):
    """
    The Game, single-player, full 2..99 rules at every curriculum stage.

    Reward:
      WIN  = 1
      LOSS = 0
      intermediate = 0

    Reset sources:
      natural:
        ordinary uniformly shuffled full game.

      reverse:
        starts from a valid late-game state taken from a generated
        guaranteed-winning trajectory. The closer target_remaining is to 98,
        the earlier the start state.

      mixed:
        mixture of natural full-game starts and teacher-generated full starts.

    The agent never observes future deck order.
    """

    metadata = {"render_modes": ["human"]}

    CARD_MIN = 2
    CARD_MAX = 99
    NUM_CARDS = 98
    HAND_SIZE = 8
    NUM_PILES = 4
    END_TURN = NUM_CARDS * NUM_PILES  # 392 -> total action count 393

    def __init__(
        self,
        reset_source: str = "natural",
        target_remaining: int = 98,
        mixed_natural_prob: float = 0.5,
        backward_teacher_prob: float = 0.30,
        render_mode: str | None = None,
        curriculum_targets: list[int] | None = None,
        curriculum_probs: list[float] | None = None,
    ):
        super().__init__()

        if reset_source not in {"natural", "reverse", "mixed"}:
            raise ValueError("reset_source must be natural, reverse, or mixed")

        if target_remaining < 2 or target_remaining > 98:
            raise ValueError("target_remaining must be in [2, 98]")

        if target_remaining % 2 != 0:
            raise ValueError(
                "target_remaining must be even in this package "
                "(teacher checkpoints are turn boundaries)"
            )

        if not 0.0 <= mixed_natural_prob <= 1.0:
            raise ValueError("mixed_natural_prob must be between 0 and 1")

        if (curriculum_targets is None) != (curriculum_probs is None):
            raise ValueError("curriculum_targets and curriculum_probs must be provided together")
        self.curriculum_targets = None
        self.curriculum_probs = None
        if curriculum_targets is not None:
            self.curriculum_targets, self.curriculum_probs = validate_distribution(
                curriculum_targets, curriculum_probs
            )
        self.sampled_target_remaining = int(target_remaining)

        self.reset_source = reset_source
        self.target_remaining = int(target_remaining)
        self.mixed_natural_prob = float(mixed_natural_prob)
        self.backward_teacher_prob = float(backward_teacher_prob)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.END_TURN + 1)

        # 98 card slots * [in_hand, unseen_in_deck, already_played]
        # + four pile tops
        # + number played this turn
        # + deck size
        self.obs_dim = self.NUM_CARDS * 3 + 4 + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.deck: list[int] = []  # front item is next draw; order is hidden
        self.hand: set[int] = set()
        self.played: set[int] = set()
        self.piles = np.array([1, 1, 100, 100], dtype=np.int16)

        self.played_this_turn = 0
        self.terminated = False
        self.won = False

        # Diagnostics count only what the agent does after reset.
        self.agent_cards_played = 0
        self.agent_turn_lengths: list[int] = []

        self.actual_reset_source = reset_source

    @staticmethod
    def _card_index(card: int) -> int:
        return card - 2

    @classmethod
    def encode_action(cls, card: int, pile: int) -> int:
        return cls._card_index(card) * cls.NUM_PILES + pile

    @classmethod
    def decode_action(cls, action: int) -> tuple[int, int]:
        card_idx, pile = divmod(int(action), cls.NUM_PILES)
        return card_idx + 2, pile

    def _clear_state(self):
        self.deck = []
        self.hand = set()
        self.played = set()
        self.piles = np.array([1, 1, 100, 100], dtype=np.int16)

        self.played_this_turn = 0
        self.terminated = False
        self.won = False

        self.agent_cards_played = 0
        self.agent_turn_lengths = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._clear_state()

        source = self.reset_source
        if source == "mixed":
            if self.np_random.random() < self.mixed_natural_prob:
                source = "natural"
            else:
                source = "reverse"

        self.actual_reset_source = source

        if source == "natural":
            self.sampled_target_remaining = 98
            self._reset_natural()
        else:
            self.sampled_target_remaining = (
                int(self.np_random.choice(self.curriculum_targets, p=self.curriculum_probs))
                if self.curriculum_targets is not None else self.target_remaining
            )
            self._reset_reverse(self.sampled_target_remaining)

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # Natural full-game start
    # ------------------------------------------------------------------

    def _reset_natural(self):
        cards = np.arange(self.CARD_MIN, self.CARD_MAX + 1, dtype=np.int16)
        self.np_random.shuffle(cards)

        # List front = next draw.
        self.deck = [int(x) for x in cards.tolist()]
        self._refill_hand()

    # ------------------------------------------------------------------
    # Reverse-curriculum start-state generator
    # ------------------------------------------------------------------

    def _inject_backward_tricks(
        self,
        seq: list[int],
        ascending: bool,
    ) -> list[int]:
        """
        Start from a monotone legal pile sequence and occasionally transform
        a +/-10 segment into one that contains an explicit backward trick.

        Ascending example:
            34, 37, 41, 44
        may become
            44, 34, 37, 41
        where 44 -> 34 is exactly -10.

        Descending is symmetric.

        The transformed sequence remains legal for that pile.
        """
        seq = list(seq)
        i = 0

        while i < len(seq) - 1:
            start = seq[i]
            target = start + 10 if ascending else start - 10

            j = None
            for k in range(i + 1, len(seq)):
                if seq[k] == target:
                    j = k
                    break

            if (
                j is not None
                and self.np_random.random() < self.backward_teacher_prob
            ):
                endpoint = seq[j]
                seq[i:j + 1] = [endpoint] + seq[i:j]
                i = j + 1
            else:
                i += 1

        return seq

    def _generate_winning_plan(self) -> list[tuple[int, int]]:
        """
        Generate a complete legal winning play plan for all 98 cards.

        Construction:
          1. randomly assign every card to one of four piles;
          2. ascending piles are ordered ascending;
          3. descending piles are ordered descending;
          4. optionally insert legal +/-10 backward-trick motifs;
          5. randomly interleave the four pile sequences while preserving
             the legal order inside each pile.

        This creates diverse, reachable, guaranteed-winning trajectories.
        """
        per_pile = {0: [], 1: [], 2: [], 3: []}

        for card in range(self.CARD_MIN, self.CARD_MAX + 1):
            pile = int(self.np_random.integers(0, 4))
            per_pile[pile].append(card)

        for pile in (0, 1):
            per_pile[pile].sort()
            per_pile[pile] = self._inject_backward_tricks(
                per_pile[pile], ascending=True
            )

        for pile in (2, 3):
            per_pile[pile].sort(reverse=True)
            per_pile[pile] = self._inject_backward_tricks(
                per_pile[pile], ascending=False
            )

        positions = [0, 0, 0, 0]
        plan: list[tuple[int, int]] = []

        while len(plan) < self.NUM_CARDS:
            available = [
                p for p in range(4)
                if positions[p] < len(per_pile[p])
            ]

            # Weight by remaining cards so interleaving is reasonably mixed.
            weights = np.array(
                [
                    len(per_pile[p]) - positions[p]
                    for p in available
                ],
                dtype=np.float64,
            )
            weights /= weights.sum()

            pile = int(self.np_random.choice(available, p=weights))
            card = per_pile[pile][positions[pile]]
            positions[pile] += 1
            plan.append((card, pile))

        return plan

    def _teacher_action_is_legal(self, card: int, pile: int) -> bool:
        return card in self.hand and self._legal_on_pile(card, pile)

    def _reset_reverse(self, target_remaining: int):
        """
        Build a guaranteed-winning full-game trajectory, then fast-forward
        along it until only target_remaining cards are still unplayed.

        The resulting state:
          - obeys the original 2..99 rules,
          - is reachable by a legitimate deck order,
          - is at a turn boundary,
          - still has at least one known winning continuation.

        The continuation is NOT exposed to the agent.
        """
        plan = self._generate_winning_plan()
        draw_order = [card for card, _ in plan]

        self.deck = list(draw_order)
        self._refill_hand()

        cards_to_fast_forward = self.NUM_CARDS - target_remaining

        if cards_to_fast_forward % 2 != 0:
            raise RuntimeError("reverse checkpoint must be at a 2-card turn boundary")

        plan_pos = 0

        while plan_pos < cards_to_fast_forward:
            # Teacher uses exactly two cards per turn.
            for _ in range(2):
                card, pile = plan[plan_pos]

                if not self._teacher_action_is_legal(card, pile):
                    raise RuntimeError(
                        f"Internal teacher-plan error: card={card}, pile={pile}"
                    )

                self.hand.remove(card)
                self.played.add(card)
                self.piles[pile] = card
                plan_pos += 1

            # End teacher turn and refill.
            self._refill_hand()

        # Agent always starts at a clean turn boundary.
        self.played_this_turn = 0

        # Do not count teacher history in agent diagnostics.
        self.agent_cards_played = 0
        self.agent_turn_lengths = []

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _refill_hand(self):
        while len(self.hand) < self.HAND_SIZE and self.deck:
            # Front is next draw.
            self.hand.add(self.deck.pop(0))

    def _minimum_turn_play(self) -> int:
        return 2 if self.deck else 1

    def _legal_on_pile(self, card: int, pile: int) -> bool:
        top = int(self.piles[pile])

        if pile < 2:  # ascending
            return card > top or card == top - 10

        # descending
        return card < top or card == top + 10

    def _legal_card_actions(self) -> list[int]:
        actions = []

        for card in self.hand:
            base = self._card_index(card) * self.NUM_PILES

            for pile in range(self.NUM_PILES):
                if self._legal_on_pile(card, pile):
                    actions.append(base + pile)

        return actions

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.terminated:
            mask[self.END_TURN] = True
            return mask

        legal_cards = self._legal_card_actions()

        for action in legal_cards:
            mask[action] = True

        if self.played_this_turn >= self._minimum_turn_play():
            mask[self.END_TURN] = True
        elif not legal_cards:
            # Forced terminal action so the action mask is never all-false.
            mask[self.END_TURN] = True

        return mask

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Card channel 1: in hand
        for card in self.hand:
            obs[self._card_index(card)] = 1.0

        # Card channel 2: unseen / still in deck.
        # Only membership is exposed, never order.
        deck_offset = self.NUM_CARDS
        for card in self.deck:
            obs[deck_offset + self._card_index(card)] = 1.0

        # Card channel 3: already played
        played_offset = 2 * self.NUM_CARDS
        for card in self.played:
            obs[played_offset + self._card_index(card)] = 1.0

        offset = 3 * self.NUM_CARDS

        obs[offset:offset + 4] = self.piles.astype(np.float32) / 100.0
        obs[offset + 4] = min(
            self.played_this_turn,
            self.HAND_SIZE,
        ) / self.HAND_SIZE
        obs[offset + 5] = len(self.deck) / self.NUM_CARDS

        return obs

    def _remaining_cards(self) -> int:
        return len(self.deck) + len(self.hand)

    # ------------------------------------------------------------------
    # Step / reward
    # ------------------------------------------------------------------

    def _finish(self, won: bool):
        self.terminated = True
        self.won = bool(won)

        # The ONLY objective.
        reward = 1.0 if won else 0.0

        return self._get_obs(), reward, True, False, self._get_info()

    def step(self, action):
        if self.terminated:
            raise RuntimeError("Episode already finished. Call reset().")

        action = int(action)
        mask = self.action_masks()

        if (
            action < 0
            or action >= self.action_space.n
            or not mask[action]
        ):
            # MaskablePPO should never do this.
            # Still: no shaped penalty, just a loss.
            self.terminated = True
            self.won = False
            info = self._get_info()
            info["illegal_action"] = action
            return self._get_obs(), 0.0, True, False, info

        if action == self.END_TURN:
            min_play = self._minimum_turn_play()

            # Forced END because player is stuck before satisfying the turn.
            if self.played_this_turn < min_play:
                return self._finish(False)

            self.agent_turn_lengths.append(self.played_this_turn)
            self.played_this_turn = 0
            self._refill_hand()

            if not self.hand and not self.deck:
                return self._finish(True)

            # Fresh turn with no legal card at all.
            if not self._legal_card_actions():
                return self._finish(False)

            return self._get_obs(), 0.0, False, False, self._get_info()

        card, pile = self.decode_action(action)

        self.hand.remove(card)
        self.played.add(card)
        self.piles[pile] = card
        self.played_this_turn += 1
        self.agent_cards_played += 1

        if not self.hand and not self.deck:
            if self.played_this_turn:
                self.agent_turn_lengths.append(self.played_this_turn)
            return self._finish(True)

        # Cannot satisfy this turn's minimum anymore.
        if (
            self.played_this_turn < self._minimum_turn_play()
            and not self._legal_card_actions()
        ):
            return self._finish(False)

        return self._get_obs(), 0.0, False, False, self._get_info()

    def _get_info(self):
        return {
            "won": bool(self.won),

            # Diagnostics only. Never used in reward.
            "remaining_cards": int(self._remaining_cards()),
            "agent_cards_played": int(self.agent_cards_played),

            "deck_size": int(len(self.deck)),
            "hand_size": int(len(self.hand)),
            "played_this_turn": int(self.played_this_turn),

            "reset_source": self.actual_reset_source,
            "target_remaining": int(self.target_remaining),
            "sampled_target_remaining": int(self.sampled_target_remaining),

            "average_agent_turn_length": (
                float(np.mean(self.agent_turn_lengths))
                if self.agent_turn_lengths
                else 0.0
            ),
        }

    def render(self):
        print(
            f"source={self.actual_reset_source} "
            f"target={self.target_remaining} "
            f"piles={self.piles.tolist()} "
            f"hand={sorted(self.hand)} "
            f"deck={len(self.deck)} "
            f"turn_played={self.played_this_turn}"
        )
