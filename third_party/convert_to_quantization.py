
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

        ### progressive training on
        progressive = getattr(quantization, "progressive", None)
        if progressive is not None and progressive.enable:
            for item in progressive.scope:
                if len(item) == 0:
                    continue
                attrs = item.split('.')
                cur = model
                find = True
                for attr in attrs:
                    if hasattr(cur, attr):
                        cur = getattr(cur, attr)
                    else:
                        find = False
                if find:
                    for m in cur.modules(): ## add to quantization list
                        if hasattr(m, 'convert_to_quantization_version'):
                            index = index + 1
                            m.convert_to_quantization_version(quantization, index)
                            #print('quantize layer {}'.format(item))
                    for m in cur.modules(): ## add to progressive training list
                        if hasattr(m, 'update_quantization'):
                            m.update_quantization(
                                    progressive=True,
                                    fm_bit=progressive.fm_start_bit,
                                    wt_bit=progressive.wt_start_bit
                                    )
                            #print('update_quantization layer {}'.format(item))
        ### progressive training off

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

    # enable probe on
    if quantization is not None and 'probe' in quantization.keyword:
        cur = model
        global_buffer = dict()
        global_buffer['index'] = 0
        global_buffer['prefix'] = 'log/probe-'
        global_buffer['verbose'] = verbose
        if not os.path.exists('log'):
            os.mkdir('log')
        for name, m in cur.named_modules():
            probe_enable = False
            for i in quantization.probe_list:
                if i in name:
                    probe_enable = True

            if probe_enable:
            #if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.Conv2d)):
                m.register_forward_hook(forward_hook)
                m.name = name
                m.global_buffer = global_buffer
                verbose('register forward_hook for module {}'.format(m.name))
    # enable probe off


