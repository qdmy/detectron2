
def convert2quantization(model, cfg):
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
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                else:
                    find = False
            if find:
                for m in cur.modules():
                    if hasattr(m, 'convert_to_quantization_version'):
                        index = index + 1
                        m.convert_to_quantization_version(quantization, index)
                        #print('quantize layer {}'.format(item))

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
    # quantization off
