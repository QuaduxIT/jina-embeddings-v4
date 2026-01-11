# Copyright © 2025-2026 Quadux IT GmbH
#    ____                  __              __________   ______          __    __  __
#   / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
#  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
# / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
# \___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/
# License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
# Author: Walter Hoffmann

"""
Download and pre-cache Jina Embeddings v4 model for offline usage.
Run this during Docker build to make the container fully offline-capable.
"""
import os
import sys
import warnings

# Suppress the Mistral regex warning (auto-fixed in newer transformers)
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")

# Set cache directories for full offline mode
os.environ['HF_HOME'] = '/app/models'
os.environ['TRANSFORMERS_CACHE'] = '/app/models'
os.environ['HF_MODULES_CACHE'] = '/app/models/modules'
os.environ['HUGGINGFACE_HUB_ALLOW_CODE_EVERYWHERE'] = '1'

# Ensure modules directory exists
os.makedirs('/app/models/modules', exist_ok=True)

from huggingface_hub import snapshot_download

model_name = 'jinaai/jina-embeddings-v4'
print(f'Downloading {model_name} to HF cache...')

# Download all files to HF cache
snapshot_download(repo_id=model_name)

print('Download complete!')
print('Pre-loading components to cache dynamic modules...')

# ============================================================================
# Import hook to patch ROPE_INIT_FUNCTIONS in dynamic modules
# ============================================================================
import importlib
import importlib.abc


class DynamicModulePatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook that patches Jina/Qwen dynamic modules as they're loaded."""
    
    def __init__(self):
        self._patched = set()
    
    def find_module(self, fullname, path=None):
        if "qwen2_5_vl" in fullname and "jina" in fullname.lower():
            return self
        return None
    
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        sys.meta_path = [p for p in sys.meta_path if p is not self]
        try:
            module = importlib.import_module(fullname)
        finally:
            sys.meta_path.insert(0, self)
        
        self._patch_module(module, fullname)
        return module
    
    def _patch_module(self, module, fullname):
        if fullname in self._patched:
            return
        self._patched.add(fullname)
        
        if hasattr(module, "ROPE_INIT_FUNCTIONS"):
            rope = module.ROPE_INIT_FUNCTIONS
            if "default" not in rope:
                if "base" in rope:
                    rope["default"] = rope["base"]
                    print(f"Patched ROPE_INIT_FUNCTIONS['default'] in {fullname}")
                elif len(rope) > 0:
                    first_key = next(iter(rope.keys()))
                    rope["default"] = rope[first_key]
                    print(f"Patched ROPE_INIT_FUNCTIONS['default'] = {first_key} in {fullname}")


# Install the import hook
_patcher = DynamicModulePatcher()
sys.meta_path.insert(0, _patcher)

# ============================================================================
# Apply patches to transformers
# ============================================================================
from transformers import modeling_utils, processing_utils

# Patch tied_weights helper
_orig_get_tied = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

def _safe_get_tied(self, *a, **kw):
    try:
        return _orig_get_tied(self, *a, **kw)
    except AttributeError:
        return []

modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = _safe_get_tied

# Patch mark_tied_weights_as_initialized to handle list vs dict
_orig_mark_tied = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized

def _safe_mark_tied(self, *a, **kw):
    """Handle both list and dict formats for all_tied_weights_keys."""
    if hasattr(self, 'all_tied_weights_keys'):
        keys = self.all_tied_weights_keys
        if isinstance(keys, list):
            # Convert list to dict format
            self.all_tied_weights_keys = {k: None for k in keys}
    try:
        return _orig_mark_tied(self, *a, **kw)
    except (AttributeError, TypeError):
        pass  # Silently ignore

modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

# Global tokenizer reference for processor patch
_tokenizer = None

# Patch ProcessorMixin.from_args_and_dict to inject tokenizer
_orig_from_args_and_dict = processing_utils.ProcessorMixin.from_args_and_dict.__func__

@classmethod
def _patched_from_args_and_dict(cls, args, processor_dict, **kwargs):
    """Inject tokenizer for Jina/Qwen processors."""
    global _tokenizer
    if _tokenizer is not None:
        kwargs["tokenizer"] = _tokenizer
        args = list(args)
        if len(args) == 0:
            args = [None, _tokenizer]
        elif len(args) == 1:
            args.append(_tokenizer)
        elif len(args) >= 2 and args[1] is None:
            args[1] = _tokenizer
        args = tuple(args)
    return _orig_from_args_and_dict(cls, args, processor_dict, **kwargs)

processing_utils.ProcessorMixin.from_args_and_dict = _patched_from_args_and_dict

# ============================================================================
# Pre-load all components to cache dynamic modules
# ============================================================================
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoProcessor

print('Pre-loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
_tokenizer = tokenizer  # Set global for processor patch

print('Pre-loading config...')
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)


def _fix_rope_config(cfg):
    """Fix rope_type and rope_scaling in config."""
    if hasattr(cfg, "rope_type") and cfg.rope_type == "default":
        cfg.rope_type = "base"
    if hasattr(cfg, "rope_scaling"):
        if cfg.rope_scaling is None:
            cfg.rope_scaling = {"type": "linear", "factor": 1.0}
        elif isinstance(cfg.rope_scaling, dict):
            cfg.rope_scaling.setdefault("type", "linear")
            cfg.rope_scaling.setdefault("factor", 1.0)
    if hasattr(cfg, "text_config"):
        _fix_rope_config(cfg.text_config)


_fix_rope_config(config)

print('Pre-loading model (this takes a while)...')
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)

print('Pre-loading processor...')
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
except Exception as e:
    print(f'Processor warning: {e}')

print('Model download and cache complete!')

# Verify all files are cached
print('\nVerifying cached files...')
import glob
modules_path = '/app/models/modules'
module_files = glob.glob(f'{modules_path}/**/*.py', recursive=True)
print(f'  Cached Python modules: {len(module_files)}')
for f in module_files[:10]:
    print(f'    - {os.path.basename(f)}')
if len(module_files) > 10:
    print(f'    ... and {len(module_files) - 10} more')
print('\n✅ Container is now fully offline-capable!')
