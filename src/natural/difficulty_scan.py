"""Held-out NC baseline scan; never learns or changes the model."""

def difficulty_scan(model, bank, evaluation):
    from src.evaluation.natural import evaluate_nc
    counts = bank.counts()
    return {t: dict(available={s: counts[s][t] for s in counts},
                    validation=evaluate_nc(model, bank, split='validation', bucket=t,
                                           games=evaluation['nc_games'], seed=evaluation['nc_seed']))
            for t in bank.stages}
