#!/bin/bash

# copying DAS data from our S3 bucket. Should take about 2-4 minutes to complete.
aws s3 cp --recursive --no-sign-request s3://2025-sz4d-das-workshop/ ./