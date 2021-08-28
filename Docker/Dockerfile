FROM python:3.7
LABEL maintainer "malmaud@gmail.com"
RUN pip install pandas matplotlib
RUN apt-get update && apt-get install -y curl
RUN mkdir /julia
RUN curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz | tar -C /julia --strip-components=1  -xzf -
ENV PATH "/julia/bin:$PATH"
COPY setup.jl /root
RUN julia /root/setup.jl
CMD julia
