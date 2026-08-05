from collections import deque

from app.runtime.request import Request


class BlockAllocationError(RuntimeError):
    """Raised when the KV block pool lacks sufficient capacity."""


class KVBlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive"
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        self.num_blocks = num_blocks
        self.block_size = block_size

        self._free_blocks: deque[int] = deque(
            range(num_blocks)
        )

        # physical block ID -> request ID
        self._owners: dict[int, str] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self._owners)

    @property
    def utilization(self) -> float:
        return (
            self.num_allocated_blocks
            / self.num_blocks
        )

    def blocks_required_for_tokens(
        self,
        num_tokens: int,
    ) -> int:
        if num_tokens < 0:
            raise ValueError(
                "num_tokens must not be negative"
            )

        # ceil(num_tokens / block_size)
        return (
            num_tokens
            + self.block_size
            - 1
        ) // self.block_size

    def can_allocate(
        self,
        num_blocks: int,
    ) -> bool:
        if num_blocks < 0:
            raise ValueError(
                "num_blocks must not be negative"
            )

        return num_blocks <= self.num_free_blocks

    def can_allocate_tokens(
        self,
        num_tokens: int,
    ) -> bool:
        required_blocks = (
            self.blocks_required_for_tokens(
                num_tokens
            )
        )

        return self.can_allocate(
            required_blocks
        )

    def allocate(
        self,
        request: Request,
        num_blocks: int,
    ) -> list[int]:
        if num_blocks < 0:
            raise ValueError(
                "num_blocks must not be negative"
            )

        if num_blocks == 0:
            return []

        if not self.can_allocate(num_blocks):
            raise BlockAllocationError(
                "Insufficient KV blocks: "
                f"requested={num_blocks}, "
                f"available={self.num_free_blocks}"
            )

        allocated: list[int] = []

        for _ in range(num_blocks):
            block_id = self._free_blocks.popleft()

            if block_id in self._owners:
                raise RuntimeError(
                    f"Free block {block_id} already has an owner"
                )

            self._owners[block_id] = (
                request.request_id
            )

            allocated.append(block_id)

        request.block_table.extend(allocated)

        return allocated

    
    def owner_of(
        self,
        block_id: int,
    ) -> str | None:
        if (
            block_id < 0
            or block_id >= self.num_blocks
        ):
            raise ValueError(
                f"Invalid block ID: {block_id}"
            )

        return self._owners.get(block_id)

    def ensure_capacity(
        self,
        request: Request,
        total_tokens: int,
    ) -> list[int]:
        additional_blocks = (
            self.additional_blocks_required(
                request=request,
                total_tokens=total_tokens,
            )
        )

        if additional_blocks <= 0:
            return []

        return self.allocate(
            request=request,
            num_blocks=additional_blocks,
        )

    def additional_blocks_required(
        self,
        request: Request,
        total_tokens: int,
    ) -> int:
        if total_tokens < 0:
            raise ValueError(
                "total_tokens must not be negative"
            )

        required_blocks = (
            self.blocks_required_for_tokens(
                total_tokens
            )
        )

        return max(
            0,
            required_blocks
            - len(request.block_table),
        )

    def free(
        self,
        request: Request,
    ) -> list[int]:
        if not request.block_table:
            return []

        block_ids = list(
            request.block_table
        )

        for block_id in block_ids:
            owner = self._owners.get(block_id)

            if owner is None:
                raise RuntimeError(
                    f"Block {block_id} has no owner"
                )

            if owner != request.request_id:
                raise RuntimeError(
                    f"Block {block_id} is owned by "
                    f"{owner}, not {request.request_id}"
                )

        for block_id in block_ids:
            del self._owners[block_id]
            self._free_blocks.append(block_id)

        request.block_table.clear()

        return block_ids

    def ensure_batch_capacity(
        self,
        requirements: list[
            tuple[Request, int]
        ],
    ) -> dict[str, list[int]]:

        seen_request_ids: set[str] = set()

        for request, total_tokens in requirements:
            if request.request_id in seen_request_ids:
                raise ValueError(
                    "Duplicate request in batch: "
                    f"{request.request_id}"
                )

            seen_request_ids.add(
                request.request_id
            )

        additional_by_request: list[
            tuple[Request, int]
        ] = []

        total_additional_blocks = 0

        for request, total_tokens in requirements:
            additional_blocks = (
                self.additional_blocks_required(
                    request=request,
                    total_tokens=total_tokens,
                )
            )

            additional_by_request.append(
                (
                    request,
                    additional_blocks,
                )
            )

            total_additional_blocks += (
                additional_blocks
            )

        if not self.can_allocate(
            total_additional_blocks
        ):
            raise BlockAllocationError(
                "Insufficient KV blocks for batch: "
                f"requested={total_additional_blocks}, "
                f"available={self.num_free_blocks}"
            )

        allocated_by_request: dict[
            str,
            list[int],
        ] = {}

        for request, additional_blocks in (
            additional_by_request
        ):
            newly_allocated = self.allocate(
                request=request,
                num_blocks=additional_blocks,
            )

            allocated_by_request[
                request.request_id
            ] = newly_allocated

        return allocated_by_request