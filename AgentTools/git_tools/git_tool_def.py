import os
import base64
import requests
from dotenv import load_dotenv
from .git_tool_schema import (GetRepositoryResult,ListRepositoryFilesResult,
ReadRepositoryFileResult,SearchRepositoriesResult, Repository, File)


from langchain_core.tools import tool


GITHUB_API = "https://api.github.com"

load_dotenv()



def github_request(method: str, endpoint: str, **kwargs):
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is not set."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.request(
        method,
        f"{GITHUB_API}{endpoint}",
        headers=headers,
        timeout=20,
        **kwargs,
    )

    if not response.ok:
        return {
            "ERROR": True,
            "status_code": response.status_code,
            "message": response.text,
        }

    return response.json()


@tool
def search_repositories(query: str) -> SearchRepositoriesResult:
    """
    Search GitHub repositories.

    Args:
        query: GitHub repository search query.
    """

    result = github_request(
        "GET",
        "/search/repositories",
        params={
            "q": query,
            "per_page": 10,
        },
    )

    if result.get("ERROR"):
        raise RuntimeError(result["message"])

    repositories = []

    for repo in result.get("items", []):
        repositories.append(
            Repository(
                owner=repo["owner"]["login"],
                name=repo["name"],
                full_name=repo["full_name"],
                description=repo["description"],
                private=repo["private"],
                default_branch=repo["default_branch"],
                html_url=repo["html_url"],
            )
        )

    return SearchRepositoriesResult(
        total_count=result.get("total_count", 0),
        repositories=repositories,
    )

@tool
def get_repository(owner: str, repo: str) -> GetRepositoryResult:
    """
    Get metadata about a GitHub repository.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
    """

    result = github_request(
        "GET",
        f"/repos/{owner}/{repo}",
    )

    if result.get("ERROR"):
        raise RuntimeError(result["message"])

    return GetRepositoryResult(
        full_name = result["full_name"],
        owner=result["owner"]["login"],
        name=result["name"],
        description = result["description"],
        private = result["private"],
        default_branch = result["default_branch"],
        language = result["language"],
        html_url = result["html_url"],
        size = result["size"],
        stars = result["stargazers_count"],
        forks = result["forks_count"],
    )


@tool
def list_repository_files(
    owner: str,
    repo: str,
    path: str = "",
) -> ListRepositoryFilesResult:
    """
    List files and directories inside a GitHub repository.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
        path: Directory path inside the repository.
              Use an empty string for the repository root.
    """

    endpoint = f"/repos/{owner}/{repo}/contents"

    if path:
        endpoint += f"/{path.strip('/')}"

    result = github_request(
        "GET",
        endpoint,
    )

    if isinstance(result, dict) and result.get("ERROR"):
        return result

    # A single file was requested instead of a directory.
    if isinstance(result, dict):
        raise RuntimeError(f"{path} is a file, not a directory.")

    files = []

    for item in result:
        files.append(
            File(
                name = item["name"],
                path = item["path"],
                type = item["type"],
            )
        )

    return ListRepositoryFilesResult(
        path = path or "/",
        items = files,
    )


@tool
def read_repository_file(
    owner: str,
    repo: str,
    path: str,
) -> ReadRepositoryFileResult:
    """
    Read the contents of a text file from a GitHub repository.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
        path: Path to the file inside the repository.
    """

    result = github_request(
        "GET",
        f"/repos/{owner}/{repo}/contents/{path.strip('/')}",
    )

    if result.get("ERROR"):
        return str({
            "ERROR": True,
            "message": f"Not such a file exists in the repo {repo}"
        })

    if result.get("type") != "file":
        return str({
            "ERROR": True,
            "message": f"{path} is not a file."
        })

    if result.get("encoding") != "base64":
        return str({
            "ERROR": True,
            "message": "GitHub returned an unsupported encoding."
        })

    try:
        content = base64.b64decode(
            result["content"]
        ).decode("utf-8")
    except UnicodeDecodeError:
        return str({
            "ERROR": True,
            "message": "File is not valid UTF-8 text."
        })

    return ReadRepositoryFileResult(
        path = result["path"],
        size = result["size"],
        content = content,
    )