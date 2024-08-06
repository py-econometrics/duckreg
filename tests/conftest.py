import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--force-regen",
        action="store_true",
        default=False,
        help="Force regeneration of test data",
    )


@pytest.fixture(scope="session")
def force_regen(request):
    return request.config.getoption("--force-regen")
