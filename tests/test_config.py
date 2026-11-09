from config import Config


# 1. Test default configuration
cfg = Config()

print("DEFAULT CONFIGURATION")
print(cfg.to_dict())


# 2. Test chunk configuration
print("\nCHUNK CONFIGURATION")
print("Minimum:", cfg.chunk.min_words)
print("Maximum:", cfg.chunk.max_words)
print("Overlap:", cfg.chunk.overlap_words)
print("Single chunk limit:", cfg.chunk.single_chunk_max_words)


# 3. Test serialization
data = cfg.to_dict()
new_cfg = Config.from_dict(data)

print("\nSERIALIZATION TEST")
print(new_cfg == cfg)


# 4. Test custom configuration
custom_cfg = Config()
custom_cfg.parser.source_preference = "json"
custom_cfg.chunk.max_words = 400
custom_cfg.preprocess.stem = False

print("\nCUSTOM CONFIGURATION")
print(custom_cfg.to_dict())