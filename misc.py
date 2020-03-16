ph = 30
hist = 180
freq = 5
day_len = 1440
cv = 4
seed = 0

path = "./GlucosePredictionATL/"

datasets_subjects_dict = {
    "IDIAB": ["1", "2", "3", "4", "5"],
    "Ohio": ["559", "563", "570", "575", "588", "591"]
}

hist_freq = hist // freq
ph_freq = ph // freq
day_len_freq = day_len // freq
