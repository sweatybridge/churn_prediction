// Refer to https://docs.basis-ai.com/getting-started/writing-files/bedrock.hcl for more details.
version = "1.0"

train {
    step train {
        image = "python:3.8.6"
        install = ["pip3 install torchvision==0.8.1"]
        script = [{sh = ["python3 upload.py"]}]
        resources {
            cpu = "1"
            memory = "4G"
        }
    }
}

serve {
    image = "basisai/express-flask:v0.0.3-gpu"
    install = [
        "pip install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "mv histogram.prom /artefact/",
            "/app/entrypoint.sh"
        ]}
    ]

    parameters {
        BEDROCK_SERVER = "serve"
    }
}
