"""Provenance capture returns a commit hash and a boolean dirty flag."""

from calcinet.framework.provenance import git_revision


def test_git_revision_shape():
    rev = git_revision()
    assert set(rev) == {"git_commit", "git_dirty"}
    assert isinstance(rev["git_commit"], str)
    assert isinstance(rev["git_dirty"], bool)


def test_git_revision_returns_a_hash_in_a_checkout():
    """Inside this repo the commit is a 40-char hex sha, not the fallback."""
    rev = git_revision()
    commit = rev["git_commit"]
    assert commit == "unknown" or (
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    )


def test_git_revision_falls_back_outside_a_repo(tmp_path):
    """Never raises just because provenance cannot be determined."""
    rev = git_revision(repo_dir=tmp_path)
    assert rev == {"git_commit": "unknown", "git_dirty": False}
