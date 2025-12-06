"""SageMaker-specific configuration constants."""


class SageMakerEnvVars:
    """SageMaker environment variable names."""

    CUSTOM_SCRIPT_FILENAME = "CUSTOM_SCRIPT_FILENAME"
    SAGEMAKER_MODEL_PATH = "SAGEMAKER_MODEL_PATH"


class SageMakerDefaults:
    """SageMaker default values."""

    SCRIPT_FILENAME = "model.py"
    SCRIPT_PATH = "/opt/ml/model/"


# SageMaker environment variable configuration mapping
SAGEMAKER_ENV_CONFIG = {
    SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: {
        "default": SageMakerDefaults.SCRIPT_FILENAME,
        "description": "Custom script filename to load (default: model.py)",
    },
    SageMakerEnvVars.SAGEMAKER_MODEL_PATH: {
        "default": SageMakerDefaults.SCRIPT_PATH,
        "description": "SageMaker model path directory (default: /opt/ml/model/)",
    },
}
