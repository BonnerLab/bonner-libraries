IDENTIFIER = "chang2019.bold5000"
N_SUBJECTS = 4
N_SESSIONS = (15, 15, 15, 9)
ROIS = (
    "EarlyVis",
    "LOC",
    "OPA",
    "PPA",
    "RSC",
)

FIGSHARE_API_BASE_URL = "https://api.figshare.com/v2"
FIGSHARE_ARTICLE_ID_V2 = 14456124
FIGSHARE_ARTICLE_ID_V1 = 6459449
URL_IMAGES = "https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"
S3_ROI_MASKS = "s3://openneuro.org/ds001499/derivatives/spm"


def get_betas_filename(subject: int, session: int) -> str:
    return (
        f"CSI{subject + 1}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-{session + 1:02}.nii.gz"
    )


def get_brain_mask_filename(subject: int) -> str:
    return f"CSI{subject + 1}_brainmask.nii.gz"


def get_imagenames_filename(subject: int) -> str:
    return f"CSI{subject + 1}_imgnames.txt"
