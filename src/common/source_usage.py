"""Shared raw usage and completed-episode diagnostics for natural experiments."""
import gymnasium as gym

def merge_counts(items):
    result = {}
    for counts in items:
        for key, value in counts.items():
            result[key] = result.get(key, 0)+value
    return result


def source_usage(env):
    if not hasattr(env, 'get_attr'):
        return {}
    return {name: merge_counts(env.get_attr('source_'+name))
            for name in ('episodes', 'transitions', 'completed', 'completed_transitions', 'wins')}


def add_usage(*items):
    return {key: merge_counts(item.get(key, {}) for item in items)
            for key in ('episodes', 'transitions', 'completed', 'completed_transitions', 'wins')}


def subtract_usage(current, previous):
    return {k: {s: count-previous.get(k, {}).get(s, 0) for s, count in values.items()
                if count != previous.get(k, {}).get(s, 0)} for k, values in current.items()}


def usage_summary(raw):
    def shares(counts):
        total = sum(counts.values())
        return {s: v/total if total else 0. for s, v in counts.items()}
    completed = raw.get('completed', {})
    transitions = raw.get('completed_transitions', {})
    sources = set(completed) | set(transitions) | set(raw.get('wins', {})) | set(raw.get('transitions', {}))
    return dict(sources={s: dict(episodes=completed.get(s, 0), transitions=raw.get('transitions', {}).get(s, 0),
                                completed_transitions=transitions.get(s, 0),
                                terminal_wins=raw.get('wins', {}).get(s, 0),
                                win_rate=raw.get('wins', {}).get(s, 0)/completed[s] if completed.get(s) else None)
                         for s in sorted(sources)},
                episode_shares=shares(completed), transition_shares=shares(transitions),
                all_transition_shares=shares(raw.get('transitions', {})))


class SourceUsageWrapper(gym.Wrapper):
    """Diagnostics only; counts starts and transitions without altering rewards."""
    def __init__(self, env):
        super().__init__(env)
        self.source_episodes = {}
        self.source_transitions = {}
        self._episode_source = None
        self.source_completed = {}
        self.source_completed_transitions = {}
        self.source_wins = {}
        self._episode_length = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._episode_length = 0
        self._episode_source = info['reset_source']
        self.source_episodes[self._episode_source] = self.source_episodes.get(self._episode_source, 0)+1
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        source = self._episode_source
        self.source_transitions[source] = self.source_transitions.get(source, 0)+1
        self._episode_length += 1
        _, reward, done, truncated, info = result
        if truncated or reward != float(done and info['won']):
            raise RuntimeError('sparse reward/WIN contract mismatch')
        if done:
            self.source_completed[source] = self.source_completed.get(source, 0)+1
            self.source_completed_transitions[source] = self.source_completed_transitions.get(source, 0)+self._episode_length
            self.source_wins[source] = self.source_wins.get(source, 0)+int(info['won'])
        return result
