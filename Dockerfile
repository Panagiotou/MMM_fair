# 1. Use Python 3.11 as the base image (lightweight)
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app  # This will be created automatically

# 3. Copy dependency file (pyproject.toml) into the container
COPY pyproject.toml ./

# 4. Install required Python dependencies
RUN pip install --no-cache-dir .

# 5. Copy the entire `mmm_fair` directory into the container
COPY mmm_fair /app/mmm_fair

# 6. Set the working directory to the `mmm_fair` directory inside the container
WORKDIR /app/mmm_fair

# 7. Run the script when the container starts
CMD ["python3", "train_and_deploy.py", "--constraint", "DP", "--classifier", "MMM_Fair_GBT", "--dataset", "kdd", "--prots", "ASEX", "AAGE", "--nprotgs", " Male", "30_60", "--moo_vis", "True", "--test", "0.3"]
