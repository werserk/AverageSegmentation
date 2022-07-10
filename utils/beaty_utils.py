def pprint(metrics):
    keys = list(metrics.keys())
    max_len = len(max(keys, key=len))
    for key in keys:
        print(f'{key.rjust(max_len, " ")}: {metrics[key]:.4f}')
