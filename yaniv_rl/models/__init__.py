''' Register rule-based models or pre-trianed models
'''

from rlcard.models.registration import register, load

register(
    model_id='yaniv-novice-rule',
    entry_point='yaniv_rl.models.yaniv_rule_models:YanivNoviceRuleModel')
