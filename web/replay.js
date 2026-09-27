'use strict';
(() => {
  const data = JSON.parse(document.getElementById('replay-data').textContent);
  const $ = id => document.getElementById(id);
  const pad = (n, size = 2) => String(n).padStart(size, '0');
  const roman = ['I', 'II', 'III', 'IV'];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let game = data.games[0], index = 0, playing = false, timer = null;
  let sound = false, audioContext = null, animationToken = 0;
  const flights = new Set();

  function cardHTML(value, direction = '✦') {
    return `<span class="card-corner">${value}<small>${direction}</small></span><span class="card-value">${value}</span><span class="card-glyph">${direction} ${direction} ${direction}</span><span class="card-corner bottom">${value}<small>${direction}</small></span>`;
  }
  function soundEffect(backward = false) {
    if (!sound) return;
    try {
      audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
      audioContext.resume().catch(() => {});
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.connect(gain); gain.connect(audioContext.destination);
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(backward ? 700 : 280, audioContext.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(backward ? 1100 : 120, audioContext.currentTime + .09);
      gain.gain.setValueAtTime(.045, audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + .13);
      oscillator.start(); oscillator.stop(audioContext.currentTime + .14);
    } catch (_) { sound = false; updateSound(); }
  }
  function updateSound() {
    $('sound').setAttribute('aria-pressed', String(sound));
    $('sound').querySelector('span').textContent = sound ? '소리 켜짐' : '소리 꺼짐';
    $('sound').title = sound ? '카드 효과음 끄기' : '카드 효과음 켜기';
  }
  function moveDescription(move) {
    if (!move) return '테이블 준비';
    if (move.kind === 'end') return move.drawn.length ? `턴 종료 · ${move.drawn.length}장 보충` : '턴 종료';
    return `${move.card} → ${roman[move.pile]}번 더미`;
  }
  function populateJournal() {
    $('journal').replaceChildren();
    game.frames.forEach((frame, step) => {
      const row = document.createElement('button');
      row.className = 'journal-row';
      row.dataset.step = step;
      const move = frame.move;
      const detail = !move ? (game.source === 'natural' ? '무작위 전체 게임 시작' : `${game.target}장 남은 후반 상태 시작`) : move.kind === 'play' ? `${move.before} ${move.pile < 2 ? '↗' : '↘'} ${move.card} · ${move.backward ? '10 차이 되돌리기' : (move.pile < 2 ? '오름차순' : '내림차순')}` : move.drawn.length ? `새 손패 + ${move.drawn.join(', ')}` : frame.state.done && !frame.state.won ? '필수 장수를 낼 수 없음' : '덱이 비어 있습니다';
      row.innerHTML = `<span class="move-number">${pad(step, 3)}<small>T${pad(move ? move.turn : 1)}</small></span><span class="move-text">${moveDescription(move)}<small>${detail}</small></span><span class="move-tag">${move?.backward ? '↶ 10' : !move ? 'START' : move.kind === 'end' ? '↵' : '↗'}</span>`;
      row.setAttribute('aria-label', `${step}수: ${moveDescription(move)}`);
      row.addEventListener('click', () => { pause(); seek(step); });
      $('journal').append(row);
    });
  }
  function messageFor(frame) {
    const state = frame.state, move = frame.move;
    if (state.done) return state.won ? 'TABLE CLEARED — 마지막 한 장까지, 완승입니다.' : `GAME OVER — 더 이상 진행할 수 없습니다. ${state.hand.length + state.deck}장이 남았습니다.`;
    if (!move) return game.source === 'natural' ? '테이블이 준비됐습니다. 첫 수를 만나보세요.' : `${game.target}장 남은 후반 상태. 여기서부터 모델의 플레이가 시작됩니다.`;
    if (move.kind === 'end') return move.drawn.length ? `TURN ${pad(state.turn)} — ${move.drawn.length}장을 보충하고, 다음 선택을 준비합니다.` : `TURN ${pad(state.turn)} — 남은 손패로 계속합니다.`;
    return move.backward ? `↶ BACKWARD 10 — ${move.before}에서 ${move.card}로. 흐름을 되돌리는 한 수.` : `${move.card} 카드를 ${roman[move.pile]}번 ${move.pile < 2 ? '오름차순' : '내림차순'} 더미에 놓았습니다.`;
  }
  function render() {
    const frame = game.frames[index], state = frame.state;
    state.piles.forEach((value, pile) => {
      $(`pile-${pile}`).innerHTML = cardHTML(value, pile < 2 ? '↑' : '↓');
      $(`pile-${pile}`).setAttribute('aria-label', `${roman[pile]}번 ${pile < 2 ? '오름차순' : '내림차순'} 더미: ${value}`);
    });
    const nextMove = game.frames[index + 1]?.move;
    $('hand').replaceChildren();
    state.hand.forEach((value, position) => {
      const card = document.createElement('div');
      const offset = position - (state.hand.length - 1) / 2;
      card.className = 'playing-card hand-card';
      card.dataset.card = value;
      card.style.setProperty('--tilt', `${offset * 2.2}deg`);
      card.style.setProperty('--lift', `${Math.abs(offset) * 2}px`);
      card.innerHTML = cardHTML(value);
      card.setAttribute('aria-label', `손패 ${value}${nextMove?.card === value ? ', 다음에 낼 카드' : ''}`);
      if (nextMove?.card === value) card.classList.add('up-next');
      if (frame.move?.drawn.includes(value)) card.classList.add('new-card');
      $('hand').append(card);
    });
    if (!state.hand.length) $('hand').innerHTML = '<span class="hand-empty">No cards in hand.</span>';
    $('turn').textContent = pad(state.turn);
    $('turn-played').textContent = state.turn_played;
    $('turn-min').textContent = `/ ${state.minimum}장 이상`;
    $('hand-count').textContent = state.hand.length;
    $('deck-count').textContent = state.deck;
    document.querySelector('.deck-area').classList.toggle('empty', state.deck === 0);
    $('legal-count').textContent = state.done ? 0 : state.legal.filter(a => a !== 392 || state.turn_played >= state.minimum).length;
    $('remaining').textContent = state.hand.length + state.deck;
    $('played').textContent = pad(state.played);
    $('backward-count').textContent = pad(game.frames.slice(0, index + 1).filter(f => f.move?.backward).length);
    $('progress-circle').style.strokeDashoffset = 125.664 * (1 - state.played / game.target);
    $('state-badge').textContent = state.done ? (state.won ? 'CLEARED' : 'GAME OVER') : index ? 'IN PLAY' : 'READY';
    $('table-message').textContent = messageFor(frame);
    $('table-message').classList.toggle('reverse-move', !!frame.move?.backward);
    document.body.classList.toggle('finished', state.done);
    document.body.classList.toggle('won', state.done && state.won);
    $('step-current').textContent = pad(index, 3);
    $('step-total').textContent = pad(game.frames.length - 1, 3);
    $('timeline').value = index;
    $('timeline').style.setProperty('--progress', `${index / (game.frames.length - 1) * 100}%`);
    $('move-caption').textContent = state.done ? (state.won ? 'TABLE CLEARED' : 'END OF GAME') : index ? `TURN ${pad(state.turn)} / MOVE ${pad(index, 3)}` : 'OPENING';
    $('prev').disabled = index === 0;
    $('restart').disabled = index === 0;
    $('next').disabled = index === game.frames.length - 1;
    $('last').disabled = state.done;
    for (const row of $('journal').children) {
      const step = Number(row.dataset.step);
      row.classList.toggle('future', step > index);
      row.classList.toggle('active', step === index);
      if (step === index) {
        row.setAttribute('aria-current', 'step');
        // Only scroll the journal; never move the entire document on mobile.
        const rowTop = row.offsetTop - $('journal').offsetTop;
        $('journal').scrollTop = rowTop - $('journal').clientHeight / 2 + row.clientHeight / 2;
      } else row.removeAttribute('aria-current');
    }
  }
  function cancelAnimation() {
    animationToken++;
    for (const flight of flights) { flight.getAnimations().forEach(a => a.cancel()); flight.remove(); }
    flights.clear();
  }
  function seek(step) {
    cancelAnimation();
    index = Math.max(0, Math.min(step, game.frames.length - 1));
    render(); updatePlay();
  }
  async function advance() {
    if (index >= game.frames.length - 1) { pause(); return; }
    cancelAnimation();
    const token = animationToken;
    const move = game.frames[index + 1].move;
    if (move.kind === 'play' && !reducedMotion.matches) {
      const from = $('hand').querySelector(`[data-card="${move.card}"]`);
      const target = $(`pile-${move.pile}`);
      if (from && target) {
        const start = from.getBoundingClientRect(), end = target.getBoundingClientRect();
        const flight = document.createElement('div');
        flight.className = `playing-card flying-card ${move.pile >= 2 ? 'descending' : ''}`;
        flight.innerHTML = cardHTML(move.card, move.pile < 2 ? '↑' : '↓');
        Object.assign(flight.style, {left:`${start.left}px`, top:`${start.top}px`, width:`${start.width}px`, height:`${start.height}px`});
        document.body.append(flight); flights.add(flight);
        const dx = end.left + end.width / 2 - start.left - start.width / 2;
        const dy = end.top + end.height / 2 - start.top - start.height / 2;
        const animation = flight.animate([
          {transform:'translate(0, 0) scale(1)',opacity:1},
          {transform:`translate(${dx * .5}px, ${dy * .5 - 25}px) scale(1.2) rotate(-5deg)`,offset:.5},
          {transform:`translate(${dx}px, ${dy}px) scale(${end.width/start.width},${end.height/start.height})`,opacity:1}
        ], {duration: Math.min(460, Number($('speed').value) * .55), easing:'cubic-bezier(.22,.7,.3,1)',fill:'forwards'});
        try { await animation.finished; } catch (_) { return; }
        flight.remove(); flights.delete(flight);
      }
    }
    if (token !== animationToken) return;
    index++; render(); soundEffect(move.backward);
    if (move.kind === 'play') {
      const target = $(`pile-${move.pile}`);
      target.classList.remove('pile-flash'); void target.offsetWidth; target.classList.add('pile-flash');
    }
    if (index === game.frames.length - 1) pause();
  }
  function updatePlay() {
    $('play-symbol').textContent = playing ? 'Ⅱ' : '▶';
    $('play-label').textContent = playing ? '일시정지' : index === game.frames.length - 1 ? '다시 보기' : '재생';
    $('play').setAttribute('aria-label', playing ? '일시정지' : '재생');
  }
  function pause() {
    playing = false; clearTimeout(timer); timer = null;
    cancelAnimation(); updatePlay();
  }
  async function tick() {
    if (!playing) return;
    const token = animationToken;
    await advance();
    // advance itself invalidates older work once; a later seek/pause must not schedule it.
    if (playing && animationToken === token + 1) timer = setTimeout(tick, Number($('speed').value));
  }
  function togglePlay() {
    if (playing) { pause(); return; }
    if (index === game.frames.length - 1) seek(0);
    playing = true; updatePlay(); tick();
  }
  function loadGame(id) {
    pause(); game = data.games.find(g => g.id === id); index = 0;
    $('timeline').max = game.frames.length - 1;
    $('table-subtitle').textContent = game.source === 'natural' ? 'NATURAL / FULL DECK' : `REVERSE / ${game.target} CARDS REMAINING`;
    $('session-note').textContent = game.source === 'natural' ? '무작위 덱 · 처음부터 시작' : `완승 경로의 후반 상태 · ${game.target}장부터 시작`;
    populateJournal(); render(); updatePlay();
  }
  function selectMode(mode) {
    for (const button of document.querySelectorAll('[data-mode]')) {
      const selected = button.dataset.mode === mode;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    }
    $('game-select').replaceChildren();
    data.games.filter(g => g.source === mode).forEach((g, i) => {
      const option = document.createElement('option');
      option.value = g.id; option.textContent = `${pad(i + 1)} · ${g.seed}`;
      $('game-select').append(option);
    });
    loadGame($('game-select').value);
  }
  $('reverse-label').textContent = `R${data.games.find(g => g.source === 'reverse').target}`;
  $('model-info').textContent = `${data.model} · ${new Intl.NumberFormat('en-US').format(data.timesteps)} STEPS`;
  $('model-info').title = `체크포인트 SHA-256: ${data.sha256}`;
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => selectMode(button.dataset.mode)));
  $('game-select').addEventListener('change', () => loadGame($('game-select').value));
  $('play').addEventListener('click', togglePlay);
  $('prev').addEventListener('click', () => { pause(); seek(index - 1); updatePlay(); });
  $('next').addEventListener('click', async () => { pause(); await advance(); updatePlay(); });
  $('restart').addEventListener('click', () => { pause(); seek(0); updatePlay(); });
  $('last').addEventListener('click', () => { pause(); seek(game.frames.length - 1); updatePlay(); });
  $('timeline').addEventListener('input', event => { pause(); seek(Number(event.target.value)); updatePlay(); });
  $('speed').addEventListener('change', () => { if (playing) { pause(); togglePlay(); } });
  $('sound').addEventListener('click', () => { sound = !sound; updateSound(); if (sound) soundEffect(); });
  $('rules').addEventListener('click', () => { pause(); $('rules-dialog').showModal(); });
  $('close-rules').addEventListener('click', () => $('rules-dialog').close());
  $('rules-dialog').addEventListener('click', e => { if (e.target === $('rules-dialog')) { const r = e.target.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) e.target.close(); } });
  $('export').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify({model:data.model,sha256:data.sha256,timesteps:data.timesteps,...game},null,2)], {type:'application/json'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = `the-game-${game.id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.querySelector('.brand').addEventListener('click', e => { e.preventDefault(); pause(); seek(0); updatePlay(); });
  document.addEventListener('keydown', e => {
    if ($('rules-dialog').open || /^(INPUT|SELECT|BUTTON|TEXTAREA)$/.test(e.target.tagName) || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowRight') { e.preventDefault(); pause(); advance().then(updatePlay); }
    if (e.code === 'ArrowLeft') { e.preventDefault(); pause(); seek(index - 1); updatePlay(); }
    if (e.code === 'Home') { e.preventDefault(); pause(); seek(0); updatePlay(); }
    if (e.code === 'End') { e.preventDefault(); pause(); seek(game.frames.length - 1); updatePlay(); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  selectMode('natural');
})();
