"""Old checkpoints may omit new default-valued dataclass fields only."""
import copy
import pytest
import torch
from tnlm.config import Config, ModelConfig
from tnlm.runtime import full_config, resume_validator, evaluation_validator
from tnlm.checkpoint import save_checkpoint, load_checkpoint
from tnlm.evaluate import _training_descriptor

FIELDS = ('depth_scaling_base', 'depth_lr_exponent', 'depth_residual_base', 'depth_residual_exponent')

@pytest.mark.parametrize('factory', [resume_validator, evaluation_validator])
def test_default_fields_admitted_and_original_preserved(factory):
    cfg = Config()
    saved = full_config(cfg)
    for name in FIELDS:
        saved['model'].pop(name)
    saved['train'].pop('deterministic', None)
    original = copy.deepcopy(saved)
    report = factory(cfg)(saved)
    assert saved == original
    assert report['defaulted_fields'] == {**{f'model.{n}': getattr(cfg.model, n) for n in FIELDS},
                                          **({'train.deterministic': False} if hasattr(cfg.train, 'deterministic') else {})}

@pytest.mark.parametrize('factory', [resume_validator, evaluation_validator])
@pytest.mark.parametrize('field,value', [('depth_scaling_base', 4), ('depth_lr_exponent', .7),
    ('depth_residual_base', 4), ('depth_residual_exponent', .7)])
def test_absent_nondefault_rejected(factory, field, value):
    cfg = Config(); saved = full_config(cfg); saved['model'].pop(field)
    setattr(cfg.model, field, value)
    with pytest.raises(ValueError, match='config mismatch'):
        factory(cfg)(saved)

@pytest.mark.parametrize('factory', [resume_validator, evaluation_validator])
def test_recorded_values_unknown_fields_and_raw_identity_remain_strict(factory):
    cfg = Config()
    for mutate in (lambda s: s['model'].update(depth_lr_exponent=.9),
                   lambda s: s['model'].update(unknown_future_field=0),
                   lambda s: s.update(data={'context': 2048})):
        saved = full_config(cfg); mutate(saved)
        with pytest.raises(ValueError, match='config mismatch'):
            factory(cfg)(saved)
    cfg.train.deterministic = True
    saved = full_config(cfg); saved['train'].pop('deterministic')
    with pytest.raises(ValueError, match='config mismatch'):
        factory(cfg)(saved)


def test_real_checkpoint_load_records_defaults_in_analysis_descriptor(tmp_path):
    cfg = Config(); saved = full_config(cfg)
    for name in FIELDS:
        saved['model'].pop(name)
    model = torch.nn.Linear(2, 2)
    save_checkpoint(tmp_path/'checkpoint', model, None, None, config=saved,
                    progress={'steps': 1, 'tokens': 2}, manifest_hash='fixture')
    loaded = load_checkpoint(tmp_path/'checkpoint', torch.nn.Linear(2, 2),
                             config_validator=evaluation_validator(cfg), restore_rng=False)
    assert loaded['config'] == saved
    report = _training_descriptor(loaded)
    assert report['config_compatibility']['defaulted_fields']['model.depth_scaling_base'] is None
    assert 'depth_scaling_base' not in report['configuration']['model']
