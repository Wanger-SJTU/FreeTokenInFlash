"""The GGUF tokenizer shim: every CONTROL / USER_DEFINED token must encode atomically
at its table id.

transformers' ``convert_gguf_tokenizer`` registers only a few special tokens (on the
qwen35moe UD file: ``<|endoftext|>``/``<|im_end|>`` made it, ``<|im_start|>`` did NOT),
so the rest BPE-split and the chat template derails into garbage. The shim marks every
control/user-defined token special; tokenizers keeps existing vocab ids, so nothing
renumbers. This test builds a minimal gpt2-style GGUF and checks the guarantee.
"""

from __future__ import annotations

import pytest


def _build_tokenizer_gguf(path: str) -> None:
    import gguf

    w = gguf.GGUFWriter(path, "qwen35moe")
    w.add_string("tokenizer.ggml.model", "gpt2")
    # byte-level-ish vocab: specials + a few merges so the converter has work; ids of
    # the specials sit mid-table exactly like a real checkpoint.
    tokens = ["a", "b", "c", "Ġ", "Ġa", "Ġb", "ab",
              "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    types = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3]
    scores = [0.0] * len(tokens)
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", types)
    w.add_array("tokenizer.ggml.scores", scores)
    w.add_array("tokenizer.ggml.merges", ["Ġ a", "Ġ b", "a b"])
    w.add_uint32("tokenizer.ggml.bos_token_id", 7)
    w.add_uint32("tokenizer.ggml.eos_token_id", 9)
    w.add_string("tokenizer.chat_template", "{% endif %}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()


def test_gguf_specials_encode_atomically_at_table_ids(tmp_path):
    pytest.importorskip("gguf")
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    p = str(tmp_path / "tok.gguf")
    _build_tokenizer_gguf(p)
    tok = load_gguf_tokenizer(p)
    for name, table_id in (("<|endoftext|>", 7), ("<|im_start|>", 8), ("<|im_end|>", 9)):
        ids = tok.encode(name, add_special_tokens=False)
        assert ids == [table_id], f"{name} must encode to its table id {table_id}, got {ids}"
        assert tok.convert_tokens_to_ids(name) == table_id
    # plain text still round-trips and the vocab never renumbers
    assert tok.vocab_size == 10
    assert tok.decode(tok.encode("abc", add_special_tokens=False)) == "abc"
