import getpass
import re

import kfp
import requests


def get_kfp_client(host: str, username: str, namespace: str, password: str=None):
    if password is None:
        password = getpass.getpass()

    session = requests.Session()
    response = session.get(host)
    
    # Get req parameter from response body
    req_parameter = re.search(r'req=(.*)\"\ target', response.text).group(1)

    # Log-in 
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"login": username, "password": password}
    url = f'{host}/dex/auth/ldap?req={req_parameter}'
    response = session.post(url, headers=headers, data=data)

    # Get session cookie
    session_cookie = session.cookies.get_dict()["authservice_session"]

    # Initialize kfp client
    client = kfp.Client(
        host=f"{host}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=namespace,
    )
    # FIXME: Set namespace again because the namespace is empty without it
    client._context_setting['namespace'] = namespace
    return client
