from pydantic import BaseModel


class Repository(BaseModel):
    owner: str
    name: str
    full_name: str
    description: str | None = None
    private: bool
    default_branch: str
    html_url: str


class SearchRepositoriesResult(BaseModel):
    total_count: int
    repositories: list[Repository]


class GetRepositoryResult(BaseModel):
    full_name: str
    owner: str
    name: str
    description: str
    private: bool
    default_branch: str
    language: str | None = None
    html_url: str
    size: int
    stars: int
    forks: int


class File(BaseModel):
    name: str
    path: str
    type: str


class ListRepositoryFilesResult(BaseModel):
    path: str
    items: list[File]


class ReadRepositoryFileResult(BaseModel):
    path: str
    size: int
    content: str