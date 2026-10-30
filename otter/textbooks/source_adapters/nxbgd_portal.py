"""Adapter for ordinary public NXBGDVN catalog pages only.

The bootstrap observations below were individually returned as official item
pages by the public web index on 2026-10-23. Runtime discovery follows normal
listing-page links and never calls undocumented endpoints.
"""
from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin

from ..catalog import Book

PORTAL = "https://taphuan.nxbgd.vn/tap-huan"
PUBLIC_LISTINGS = {
 10: "https://taphuan.nxbgd.vn/tap-huan/page-1?grade=10&tab=public",
 11: "https://taphuan.nxbgd.vn/tap-huan/page-1?grade=11",
 12: "https://taphuan.nxbgd.vn/tap-huan?grade=12&tab=public",
}

# These are official, human-usable item URLs observed from indexed public pages.
# No electronic file URL is included or inferred.
OBSERVED = [
 (10,"Vật lí 10","Vật lí",None,"vat-li-10-940250226.940250226","turn1search2"),
 (10,"Hóa học 10","Hóa học",None,"hoa-hoc-10-940249942.940249942","turn3search1"),
 (10,"Sinh học 10","Sinh học",None,"sinh-hoc-10-939773641.939773641","turn3search0"),
 (10,"Lịch sử 10","Lịch sử",None,"lich-su-10-940218902.940218902","turn3search6"),
 (10,"Địa lí 10","Địa lí",None,"dia-li-10-939903840.939903840","turn3search2"),
 (10,"Ngữ văn 10, tập một","Ngữ văn","tập một","ngu-van-10-tap-mot-939855181.939855181","turn1search9"),
 (10,"Ngữ văn 10, tập hai","Ngữ văn","tập hai","ngu-van-10-tap-hai-940208193.940208193","turn7search2"),
 (11,"Vật lí 11","Vật lí",None,"vat-li-11-940070814.940070814","turn4search3"),
 (11,"Hóa học 11","Hóa học",None,"hoa-hoc-11-939753998.939753998","turn4search0"),
 (11,"Sinh học 11","Sinh học",None,"sinh-hoc-11-940163795.940163795","turn4search2"),
 (11,"Lịch sử 11","Lịch sử",None,"lich-su-11-940006350.940006350","turn6search1"),
 (11,"Địa lí 11","Địa lí",None,"dia-li-11-939832163.939832163","turn6search3"),
 (11,"Ngữ văn 11, tập một","Ngữ văn","tập một","ngu-van-11-tap-mot-939696849.939696849","turn4search1"),
 (11,"Ngữ văn 11, tập hai","Ngữ văn","tập hai","ngu-van-11-tap-hai-940224592.940224592","turn6search0"),
 (12,"Vật lí 12","Vật lí",None,"vat-li-12-940172299.940172299","turn5search1"),
 (12,"Hóa học 12","Hóa học",None,"hoa-hoc-12-939998682.939998682","turn5search2"),
 (12,"Sinh học 12","Sinh học",None,"sinh-hoc-12-939869678.939869678","turn5search0"),
 (12,"Lịch sử 12","Lịch sử",None,"lich-su-12-939853322.939853322","turn7search0"),
 (12,"Địa lí 12","Địa lí",None,"dia-li-12-939943182.939943182","turn7search4"),
 (12,"Ngữ văn 12, tập một","Ngữ văn","tập một","ngu-van-12-tap-mot-940003905.940003905","turn5search3"),
 (12,"Ngữ văn 12, tập hai","Ngữ văn","tập hai","ngu-van-12-tap-hai-940165602.940165602","turn7search1"),
]

def bootstrap_catalog() -> list[Book]:
    out=[]
    for grade,title,subject,volume,path,ref in OBSERVED:
        item=f"{PORTAL}/chi-tiet-sach/{path}"
        out.append(Book.observed(title=title,subject=subject,grade=grade,item_url=item,
            evidence_url=item,original_series_label="Bộ SGK thống nhất"))
        out[-1].volume=volume
    return out

class _Links(HTMLParser):
    def __init__(self): super().__init__(); self.href=None; self.text=[]; self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag == "a": self.href=dict(attrs).get("href"); self.text=[]
    def handle_data(self, data):
        if self.href is not None: self.text.append(data)
    def handle_endtag(self, tag):
        if tag=="a" and self.href is not None:
            self.links.append((self.href," ".join("".join(self.text).split()))); self.href=None

def discover_public_listing(url: str, timeout: int = 15) -> list[tuple[str,str]]:
    """Visit an explicitly public listing and return visible book-detail links.

    Network and parse failures propagate to the caller, which must record them
    as navigation failures rather than treating them as evidence of no books.
    """
    if not url.startswith("https://taphuan.nxbgd.vn/tap-huan"):
        raise ValueError("Only official public catalog URLs are accepted")
    request=urllib.request.Request(url,headers={"User-Agent":"OTTER textbook catalog (polite public page visit)"})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        if response.status != 200: raise OSError(f"listing HTTP {response.status}")
        html=response.read(5_000_001)
        if len(html)>5_000_000: raise ValueError("listing exceeds 5 MB safety limit")
        if "text/html" not in response.headers.get("Content-Type","").lower(): raise ValueError("listing response is not HTML")
    parser=_Links(); parser.feed(html.decode("utf-8","replace"))
    return [(urljoin(url,href),label) for href,label in parser.links if "/chi-tiet-sach/" in href and label]
