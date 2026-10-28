# Fixed100 numeric control-flow fixture

**PASS, numeric fixture only.** No source TRAIN, INNER, OUTER or Test arrays were opened. Two independent fresh processes each used four generated TRAIN and four generated INNER windows, a width4/block1 CNN and 100 actual Adam updates. Validation scalars were explicitly mocked for control-flow assertions and must never be cited as model performance.

Constant validation plateau: 100 epochs/100 optimizer updates/100 saved epoch files, first-tie best checkpoints at epoch1, last checkpoint at100. Shadow ES7 would stop at8 but never stopped training. The scheduler still reduced learning rate to1e-6. Separate-minima case: best-window epoch27, best-macro epoch73, last100. Every numeric epoch checkpoint was loaded and its epoch/SHA checked; final optimizer steps and actual parameter updates were verified.

This proves runner behavior, not a global minimum or unbiased validation. Repeated INNER selection remains optimistic, and previously viewed cross panels are not blind. CPU thread changes also prevent a bitwise prefix comparison with older trajectories. Numeric weights are temporary and forbidden for production reuse.

[Detailed JSON](R6_FIXED100_NUMERIC_FIXTURE.json)
