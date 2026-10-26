# Revision bias in the value of Census retail sales

Shapley value of retail sales (95% CI, wider of firm- and quarter-clustered bootstrap) when the same backtest uses different versions of the data. Development period only.

| data | quarters | retail value | low | high | verdict | inflation vs honest |
| --- | --- | --- | --- | --- | --- | --- |
| realtime: Figures as published on the nowcast date (honest) | all | +0.030 | +0.010 | +0.044 | helps | +0.000 |
| realtime: Figures as published on the nowcast date (honest) | outside COVID | -0.003 | -0.012 | +0.005 | no effect | +0.000 |
| revised: Today's revised figures, release timing respected | all | +0.029 | +0.011 | +0.043 | helps | -0.000 |
| revised: Today's revised figures, release timing respected | outside COVID | -0.002 | -0.011 | +0.005 | no effect | +0.000 |
| revised_no_lag: Today's revised figures, including months not yet released | all | +0.059 | +0.027 | +0.090 | helps | +0.030 |
| revised_no_lag: Today's revised figures, including months not yet released | outside COVID | +0.003 | -0.008 | +0.015 | no effect | +0.006 |
