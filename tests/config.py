import ml_collections as mlc

consts = mlc.ConfigDict(
    {
        "batch_size": 1,
        "n_res": 256,
        "n_seq": 132,
        "n_templ": 4,
        "n_extra": 1024,
        "eps": 5e-4,
        # For compatibility with DeepMind's pretrained weights, it's easiest for
        # everyone if these take their real values.
        "c_m": 256,
        "c_z": 128,
        "c_s": 384,
        "c_t": 64,
        "c_e": 64,
    }
)

config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "masked_msa": {
                    "profile_prob": 0.1,
                    "same_prob": 0.1,
                    "uniform_prob": 0.1,
                },
            }
        }
    }
)
