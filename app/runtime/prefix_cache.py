from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixCacheEntry:
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]


class PrefixCache:
    def __init__(self, block_manager) -> None:
        self._block_manager = block_manager
        self._entries: dict[
            tuple[int, ...],
            PrefixCacheEntry,
        ] = {}

    def lookup(
        self,
        token_ids: list[int] | tuple[int, ...],
    ) -> PrefixCacheEntry | None:
        key = tuple(token_ids)
        return self._entries.get(key)

    def lookup_longest_prefix(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> PrefixCacheEntry | None:
        cacheable_tokens = (
            len(token_ids) // block_size
        ) * block_size

        while cacheable_tokens > 0:
            key = tuple(
                token_ids[:cacheable_tokens]
            )

            entry = self._entries.get(key)

            if entry is not None:
                return entry

            cacheable_tokens -= block_size

        return None

    def insert(
        self,
        token_ids,
        block_ids,
    ) -> None:
        key = tuple(token_ids)

        if key in self._entries:
            return

        entry = PrefixCacheEntry(
            token_ids=key,
            block_ids=tuple(block_ids),
        )

        self._block_manager.retain_blocks(
            entry.block_ids
        )

        self._entries[key] = entry

    def evict(
        self,
        token_ids,
    ) -> None:
        key = tuple(token_ids)

        entry = self._entries.pop(key, None)

        if entry is None:
            return

        self._block_manager.release_blocks(
            entry.block_ids
        )

    @staticmethod
    def get_cacheable_prefix_length(
        num_tokens: int,
        block_size: int,
    ) -> int:
        if num_tokens < 0:
            raise ValueError(
                "num_tokens must be non-negative"
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        return (
            num_tokens // block_size
        ) * block_size