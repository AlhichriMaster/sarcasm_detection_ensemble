# Build context is the project root:
#   docker build --platform linux/amd64 -f terraform/Dockerfile -t sarcasm-detection:latest .
#
# tensorflow-cpu covers all ops used by the LSTM + MultiHeadAttention +
# Transformer models without the GPU overhead.

FROM public.ecr.aws/lambda/python:3.12

# Install TensorFlow (CPU) and runtime dependencies
RUN pip install --no-cache-dir \
    "tensorflow-cpu==2.17.0" \
    "numpy==1.26.4" \
    "h5py>=3.10,<3.13" \
    "keras-preprocessing" \
    "boto3>=1.26"

# Copy the project's model source code into the Lambda task root
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Ensure Python package markers exist so imports work correctly
RUN touch ${LAMBDA_TASK_ROOT}/src/__init__.py \
          ${LAMBDA_TASK_ROOT}/src/models/__init__.py \
          ${LAMBDA_TASK_ROOT}/src/data/__init__.py

# Copy the Lambda handler (lives alongside main.tf in terraform/)
COPY terraform/lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]
