"""One Rich live panel on TTY; one compact line per evaluation otherwise."""
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class Dashboard:
    def __init__(self, console=None):
        self.console = console or Console()
        self.live = None

    def __enter__(self):
        if self.console.is_terminal:
            self.live = Live(console=self.console, refresh_per_second=2)
            self.live.start()
        return self

    def update(self, record, *, history=False):
        natural = record.get('natural_full') or {}
        metric = lambda r: f"{r.get('wins', '?')} / {r.get('games', '?')}  " + (f"{r['win_rate']:.1%}" if r.get('win_rate') is not None else '')
        table = Table.grid(padding=(0, 3))
        table.add_row('Natural full', metric(natural))
        if record.get('natural_full_large'):
            table.add_row('Natural full large', metric(record['natural_full_large']))
        table.add_row('Stage', record.get('current_stage', '?')+' → '+str(record.get('next_stage') or 'final'))
        table.add_row('Stage steps', f"{record.get('stage_steps', 0):,} / {record.get('stage_budget', 0):,}")
        table.add_row('Global steps', f"{record.get('global_steps', 0):,}")
        table.add_row('NC validation', metric(record.get('nc_current') or {}))
        table.add_row('Next NC', metric(record.get('nc_next') or {}))
        for label, key in [('Cumulative', 'usage_cumulative'), ('Window', 'usage_window')]:
            stats = record.get(key, {})
            for title, field in [('episode share', 'episode_shares'), ('transition share', 'transition_shares')]:
                shares = stats.get(field, {})
                nc, replay = shares.get('natural_conditioned', 0), shares.get('reverse', 0)
                table.add_row(f'{label} {title}', f'NC {nc:.1%} / Replay {replay:.1%}' if shares else 'No completed episodes')
            train = stats.get('sources', {}).get('natural_conditioned', {})
            table.add_row(f'{label} NC train WIN', f"{train.get('terminal_wins', 0)} / {train.get('episodes', 0)}")
        table.add_row('Promotion', record.get('status', 'WAIT'))
        table.add_row('Next evaluation', f"+{record.get('next_evaluation', 0):,} transitions")
        if self.live:
            self.live.update(Panel(table, title='Natural Reverse Curriculum'))
        if history and not self.live:
            self.console.print(f"Natural full {metric(natural)} | {record.get('current_stage')} "
                               f"steps={record.get('global_steps')} {record.get('status')}", markup=False, highlight=False)

    def __exit__(self, *args):
        if self.live:
            self.live.stop()
