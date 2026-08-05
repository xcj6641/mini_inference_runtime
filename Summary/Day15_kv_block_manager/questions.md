1.
This function:
       self.block_manager.ensure_batch_capacity(
            requirements
        )

raised exception but there are no resume logic. The system will never recover.

2.
Alternative: simply run decode before prefill?

You might reasonably ask:

Since decode is selected first, why not just execute decode first?

You could:

select decode
run decode
select prefill
run prefill

That would also eliminate this particular stale-capacity problem.

But there may be reasons your scheduler currently executes prefill first, and changing execution order affects batching behavior and latency policy.

More importantly, reservation is the more general solution.

Even if execution remains:

prefill → decode

reserved decode blocks cannot be stolen.

3. 
chunked prefill? what is this?

4. 
vLLM describes the policy as:

    1. schedule as many decoding requests as possible
    2. schedule unfinished chunked prefills
    3. schedule other work
    4. schedule new prefills

and says this allows prefill and decode requests to be batched together, improving GPU utilization.

How prefill and decode have relationship with GPU utilization?

5. 
Why do you design in this way?
like plan decode/prefill budget -> execute decode/prefill

6. some important metrics
TTFT, ITL(inter-token latency)

7. failure-recovery
If model execution throws after reservation:

reserve blocks
↓
runner.prefill_batch() raises

the blocks stay allocated because there is currently no transactional rollback.

Likewise:

reserve decode growth
↓
runner.decode_batch() raises

the newly reserved block remains attached.

Eventually a hardened runtime could do:

plan
reserve
try execution
except:
    rollback reservation
But I would not implement that now. That's failure-recovery infrastructure rather than the core inference mechanism we're learning.