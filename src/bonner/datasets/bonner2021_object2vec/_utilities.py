import numpy as np
from scipy.io import loadmat

IDENTIFIER = "bonner2021.object2vec"
N_SUBJECTS = 4
BRAIN_DIMENSIONS = (79, 95, 79)
ROIS = {
    "EVC": "scrambled",
    "LOC": "objects",
    "PFS": "objects",
    "OPA": "scenes",
    "PPA": "scenes",
    "RSC": "scenes",
    "FFA": "faces",
    "OFA": "faces",
    "STS": "faces",
    "EBA": "faces",
}

URLS = {
    "stimuli": "https://osf.io/g74ar/download",
    "conditions": "https://osf.io/8eu5h/download",  # dropbox://object_associations_follow_ups/analyses/betas007/subj00{1, 2, 3, 4}/stacked_means.mat
    "activations": (  # dropbox://object_associations_follow_ups/analyses/betas005/subj00{1, 2, 3, 4}/betas.mat
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADyAjIte6V1oME73akilq3Pa/analyses/betas005/subj001/betas.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AABcfszt5H8F-LOlmyjStiLaa/analyses/betas005/subj002/betas.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADyTe9a2PDwbzpDIpNMxoJFa/analyses/betas005/subj003/betas.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADLmGXvsgJCqG5gr_la0mfza/analyses/betas005/subj004/betas.mat?dl=1",
    ),
    "noise_ceilings": (  # dropbox://object_associations_follow_ups/analyses/betas005/subj00{1, 2, 3, 4}/noise_ceiling.mat
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACk9iYuFqHrkqdt-KVB0Ucga/analyses/betas005/subj001/noise_ceiling.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACrcp1WmO3-WEIq_Z2cKdUea/analyses/betas005/subj002/noise_ceiling.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACzrP35FbbP1Trd2eh5eMSYa/analyses/betas005/subj003/noise_ceiling.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADsMABQqHbchfKkhWEgoYw4a/analyses/betas005/subj004/noise_ceiling.mat?dl=1",
    ),
    "rois": (  # dropbox://object_associations_follow_ups/analyses/rois042/subj00{1, 2, 3, 4}/indices.mat
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAAjYK53tpBlVM6m_XLBMfsPa/analyses/rois042/subj001/indices.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AABz1nB1eA4uc4O5sLNM2Oe-a/analyses/rois042/subj002/indices.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAB-rUgw9hYsBBd6thNLI2TIa/analyses/rois042/subj003/indices.mat?dl=1",
        "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAAkEucM_Ug4Lv14V8VV3a1Ua/analyses/rois042/subj004/indices.mat?dl=1",
    ),
    "cv_sets": (  # dropbox://object_associations_follow_ups/analyses/objs003/subj00{1, 2, 3, 4}/sets.mat
        "https://osf.io/asgkn/download",
        "https://osf.io/wd9s2/download",
        "https://osf.io/8wpr7/download",
        "https://osf.io/3c5tb/download",
    ),
    "contrasts": {  # dropbox://object_associations_follow_ups/analyses/glm010/subj00{1, 2, 3, 4}/{scrambled, objects, scenes, faces}.nii
        "scrambled": (
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADAUjBlctB2DEo0YsI41tu_a/analyses/glm010/subj001/scrambled.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADT3ZgOJmcqGtRgJE2Ycq48a/analyses/glm010/subj002/scrambled.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAB2dMnhshgAp8-lj0Rd6iUCa/analyses/glm010/subj003/scrambled.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAACPIKlwJHo9pnvexGIwoKUa/analyses/glm010/subj004/scrambled.nii?dl=1",
        ),
        "objects": (
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACRprX1-5B9msoc50urwG75a/analyses/glm010/subj001/objects.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACGcnPC7QuvEeFxYnXgdFFQa/analyses/glm010/subj002/objects.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACWU5Xg4dgG_wQhR0ahKgMaa/analyses/glm010/subj003/objects.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADTXfAVlNwUjm-6fgbgoeB1a/analyses/glm010/subj004/objects.nii?dl=1",
        ),
        "scenes": (
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AADCJa0kbb5WhtcJ6949cgrza/analyses/glm010/subj001/scenes.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACKAT4ZHEaWlh9UEJYHkAHla/analyses/glm010/subj002/scenes.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AABygH40BXa8xbJNgN-d8jjwa/analyses/glm010/subj003/scenes.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAAOz3XLES6VtoWxFo81W8Zia/analyses/glm010/subj004/scenes.nii?dl=1",
        ),
        "faces": (
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAB8CmJxl1E5EFicM7K3uEE8a/analyses/glm010/subj001/faces.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AABQrlqqyhkK_2ri3IiHqrSca/analyses/glm010/subj002/faces.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AAA-f3u3cu2UZ2MONG1AGzwPa/analyses/glm010/subj003/faces.nii?dl=1",
            "https://www.dropbox.com/sh/c8opg6mczg7nsdy/AACX6Py-aWiXgfb5qfVXGeX_a/analyses/glm010/subj004/faces.nii?dl=1",
        ),
    },
}

FILENAMES = {
    "stimuli": "stimuli.zip",
    "conditions": "conditions.mat",
    "activations": [f"betas_subj{subject}.mat" for subject in range(N_SUBJECTS)],
    "noise_ceilings": [
        f"noise_ceilings_subj{subject}.mat" for subject in range(N_SUBJECTS)
    ],
    "rois": [f"rois_subj{subject}.mat" for subject in range(N_SUBJECTS)],
    "cv_sets": [f"sets_subj{subject}.mat" for subject in range(N_SUBJECTS)],
    "contrasts": {
        contrast: [
            f"contrast_{contrast}_subj{subject}.nii" for subject in range(N_SUBJECTS)
        ]
        for contrast in ("scrambled", "objects", "scenes", "faces")
    },
}


def load_conditions() -> np.ndarray:
    return loadmat(FILENAMES["conditions"], simplify_cells=True)["stacked"]["conds"]
