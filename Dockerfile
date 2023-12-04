#  Copyright (c) VKU.NewEnergy.

#  This source code is licensed under the Apache-2.0 license found in the
#  LICENSE file in the root directory of this source tree.

FROM python:3.10.10

WORKDIR /app/ai

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["python", "main.py"]