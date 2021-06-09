
import torch
import os

def forward_hook(module, input, output):
    name = module.name
    global_buffer = module.global_buffer

    if name in global_buffer:
        return

    index = global_buffer['index']
    index = index + 1
    global_buffer['index'] = index
    global_buffer[name] = 'Saved'
    prefix = global_buffer['prefix'] if 'prefix' in global_buffer else ''
    verbose = global_buffer['verbose'] if 'verbose' in global_buffer else print
    verbose("saving tensors of {} {}".format(name, index))
    torch.save(input[0], "{}{}-{}-input.pth".format(prefix, name, index))
    torch.save(output[0], "{}{}-{}-output.pth".format(prefix, name, index))

def convert2quantization(model, cfg, verbose=print):
    verbose("going to convert model to quantization version")
    # quantization on
    quantization = cfg.MODEL.QUANTIZATION if hasattr(cfg.MODEL, 'QUANTIZATION') else None
    if quantization is not None:
        assert hasattr(quantization, 'scope') and hasattr(quantization, 'keyword')
        index = -1
        for item in quantization.scope:
            if len(item) == 0:
                continue
            attrs = item.split('.')
            cur = model
            find = True
            for attr in attrs:
                if attr == "":
                    continue
                elif hasattr(cur, attr):
                    cur = getattr(cur, attr)
                else:
                    find = False
            if find:
                for m in cur.modules():
                    if hasattr(m, 'convert_to_quantization_version'):
                        index = index + 1
                        verbose('quantize layer {}, index {}'.format(item, index))
                        m.convert_to_quantization_version(quantization, index)

        ### Norm layer
        cur = model
        index = -1
        for m in cur.modules(): ## add to quantization list
            if hasattr(m, 'convert_norm_to_quantization_version'):
                index = index + 1
                m.convert_norm_to_quantization_version(quantization, index)

        ### Eltwise layer
        cur = model
        index = -1
        for m in cur.modules(): ## add to quantization list
            if hasattr(m, 'convert_eltwise_to_quantization_version'):
                index = index + 1
                m.convert_eltwise_to_quantization_version(quantization, index)
    # quantization off
    verbose("convert model to quantization version ... done")


