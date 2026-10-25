"""Validate actual serializer/processor coordinates; never infer patches from a span."""
from dataclasses import dataclass
from ..serializer import Serialized, to_list, visual_token_counts


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class Alignment:
    patches: tuple[int, ...]
    rows: tuple[int, ...]
    grid: tuple[tuple[int, ...], ...]


def align(serialized: Serialized, *, image_token_id: int, vision_start_id: int,
          vision_end_id: int, merge_size: int, video_token_id: int | None = None,
          pad_token_id: int | None = None) -> Alignment:
    s = serialized
    ids = s.input_ids
    raw_ids = [int(x) for x in to_list(s.model_inputs['input_ids'])]
    mask = [int(x) for x in to_list(s.model_inputs.get('attention_mask', [1] * len(ids)))]
    if ids != raw_ids or len(mask) != len(ids) or any(x not in (0, 1) for x in mask):
        raise AlignmentError('processor token IDs/mask differ')
    grid_raw = s.model_inputs.get('image_grid_thw')
    if grid_raw is None or merge_size < 1:
        raise AlignmentError('missing processor grid/merge')
    grid = tuple(tuple(int(v) for v in row) for row in
                 (grid_raw.tolist() if hasattr(grid_raw, 'tolist') else grid_raw))
    if len(grid) != 1 or len(grid[0]) != 3 or any(v < 1 for v in grid[0]):
        raise AlignmentError('exactly one positive image grid required')
    try:
        counts = visual_token_counts(s.model_inputs, merge_size)
    except (ValueError, RuntimeError) as exc:
        raise AlignmentError(str(exc)) from exc
    excluded = {vision_start_id, vision_end_id, video_token_id, pad_token_id} - {None}
    if image_token_id in excluded or video_token_id is not None and video_token_id in ids:
        raise AlignmentError('video or ambiguous image token ID')
    patches = tuple(i for i, token in enumerate(ids) if token == image_token_id and mask[i])
    if (len(patches) != counts[0] or len(patches) != s.num_visual_tokens or
            patches != tuple(s.image_positions) or not patches or
            patches != tuple(range(patches[0], patches[-1] + 1)) or
            patches[0] == 0 or patches[-1] + 1 >= len(ids) or
            ids[patches[0]-1] != vision_start_id or ids[patches[-1]+1] != vision_end_id):
        raise AlignmentError('image token IDs/grid/brackets disagree')
    action = s.action_token_positions
    completed = s.completed_action_row
    nonpad = [i for i, value in enumerate(mask) if value]
    if (not action or not s.decoded_action_span or not s.decoded_action_span.strip() or
            completed != action[-1] or action != sorted(set(action)) or
            action[0] <= patches[-1] or any(i < 0 or i >= len(ids) or not mask[i] for i in action) or
            not nonpad or s.final_prompt_row != nonpad[-1]):
        raise AlignmentError('invalid completed action/final prompt alignment')
    return Alignment(patches, tuple(i for i in range(completed, s.final_prompt_row + 1) if mask[i]), grid)
