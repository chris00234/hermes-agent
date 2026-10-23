"""Native turn accounting, separate from the last request's context size."""

from agent.usage_pricing import CanonicalUsage


_TOKEN_FIELDS = (
    "inputTokens", "cachedInputTokens", "cacheWriteInputTokens", "outputTokens",
    "reasoningOutputTokens", "totalTokens",
)


def record_usage(result, usage: dict) -> None:
    last = usage.get("last")
    if not isinstance(last, dict):
        return
    result.token_usage_last = dict(last)
    total = usage.get("total")
    if isinstance(total, dict):
        previous = result.token_usage_snapshot
        if previous == total:
            return
        delta = last
        if previous and total.get("totalTokens", 0) >= previous.get("totalTokens", 0):
            delta = {key: max(0, total.get(key, 0) - previous.get(key, 0)) for key in _TOKEN_FIELDS}
        result.token_usage_snapshot = dict(total)
    else:
        delta = last
    if result.token_usage_total is None:
        result.token_usage_total = {}
    for key in _TOKEN_FIELDS:
        result.token_usage_total[key] = result.token_usage_total.get(key, 0) + max(0, int(delta.get(key, 0)))
    result.api_call_count += 1


def canonical_usage(usage: dict) -> CanonicalUsage:
    cached = max(0, int(usage.get("cachedInputTokens", 0)))
    written = max(0, int(usage.get("cacheWriteInputTokens", 0)))
    return CanonicalUsage(
        input_tokens=max(0, int(usage.get("inputTokens", 0)) - cached - written),
        output_tokens=max(0, int(usage.get("outputTokens", 0))),
        cache_read_tokens=cached, cache_write_tokens=written,
        reasoning_tokens=max(0, int(usage.get("reasoningOutputTokens", 0))), raw_usage=usage,
    )


def usage_dict(usage: CanonicalUsage) -> dict:
    return {
        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens, "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens, "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens, "reasoning_tokens": usage.reasoning_tokens,
    }
