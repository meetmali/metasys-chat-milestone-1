import ssl, os, urllib3, requests
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
urllib3.disable_warnings()
_orig = requests.Session.merge_environment_settings
def _no_ssl(self, url, proxies, stream, verify, cert):
    s = _orig(self, url, proxies, stream, verify, cert)
    s['verify'] = False
    return s
requests.Session.merge_environment_settings = _no_ssl
