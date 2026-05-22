from pydantic import BaseModel, HttpUrl


class IngestFileResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class IngestURLRequest(BaseModel):
    url: HttpUrl
    title: str | None = None


class IngestURLResponse(BaseModel):
    document_id: str
    url: str
    status: str
    message: str
