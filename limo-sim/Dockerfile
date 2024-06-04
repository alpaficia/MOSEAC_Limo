# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch

RUN apt update && apt install -y ffmpeg libsm6 libxext6 libgl1*

# Set the working directory
WORKDIR /root

# Copy the current directory contents into the container
COPY . /root/MOSEAC
# Install any needed packages specified in requirements.txt
RUN cd /rooy/MOSEAC/
RUN pip3 install -r requirement.txt
# Run when the container launches
CMD ["python3", "main.py"]
