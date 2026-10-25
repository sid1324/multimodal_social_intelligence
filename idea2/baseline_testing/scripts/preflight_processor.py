"""CPU-only check of the exact video processor input format used on GPU."""
import json
import itertools
import numpy as np
from transformers import AutoProcessor

from social_rl.common import STORE, config, write_json
from social_rl.tasks import ego_prompt


def main():
    cfg = config()
    processor = AutoProcessor.from_pretrained(STORE / 'models/Qwen3.5-9B', local_files_only=True)
    frames = np.zeros((cfg['frames'], 192, 256, 3), dtype=np.uint8)
    metadata = {'fps': 2.0, 'total_num_frames': cfg['frames'],
                'frames_indices': list(range(cfg['frames']))}
    row = {'behaviors': ['A', 'B', 'C', 'D', ''],
           'justifications': ['Why A', 'Why B', 'Why C', 'Why D', '']}
    outputs = {}
    for task in ('action', 'justification', 'sensibility'):
        prompt = ego_prompt(row, task, action=2)
        chunks = [frames[i:i + 2] for i in range(0, len(frames), 2)]
        metadata_chunks = [{**metadata, 'frames_indices': metadata['frames_indices'][i:i + 2]}
                           for i in range(0, len(frames), 2)]
        content = [{'type': 'video'} for _ in chunks] + [{'type': 'text', 'text': prompt}]
        text = processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=cfg['enable_thinking'])
        batch = processor(text=[text], videos=chunks, video_metadata=metadata_chunks,
                          do_sample_frames=False, return_tensors='pt',
                          size={'shortest_edge': 128 * 32 * 32,
                                'longest_edge': cfg['frames'] * cfg['frame_max_side'] ** 2})
        assert batch['input_ids'].shape[0] == 1
        assert any('video' in key for key in batch.keys()), list(batch)
        video_runs = sum(key == 2 for key, _ in itertools.groupby(batch['mm_token_type_ids'][0].tolist()))
        assert video_runs == len(batch['video_grid_thw']) == len(chunks)
        outputs[task] = {'input_tokens': batch['input_ids'].shape[1],
                         'keys': list(batch.keys()), 'video_runs': video_runs,
                         'grid_rows': len(batch['video_grid_thw'])}
    write_json(STORE / 'runs/processor_preflight.json', outputs)
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
