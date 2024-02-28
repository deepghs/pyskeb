from functools import lru_cache

from random_user_agent.params import SoftwareName, OperatingSystem
from random_user_agent.user_agent import UserAgent


@lru_cache()
def _ua_pool():
    software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, SoftwareName.EDGE.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.MACOS.value]

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=1000)
    return user_agent_rotator


def get_random_ua():
    return _ua_pool().get_random_user_agent()


@lru_cache()
def _ua_mobile_pool():
    software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, SoftwareName.SAFARI.value]
    operating_systems = [OperatingSystem.ANDROID.value, OperatingSystem.IOS.value]

    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=1000)
    return user_agent_rotator


def get_random_mobile_ua():
    return _ua_mobile_pool().get_random_user_agent()
