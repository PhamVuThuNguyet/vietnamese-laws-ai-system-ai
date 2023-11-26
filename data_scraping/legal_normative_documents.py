import json

import requests
from bs4 import BeautifulSoup


def get_all_links():
    with open("links.txt", "r+", encoding="utf-8") as f:
        links = json.load(f)

    start = len(links) // 30 + 1
    stop = 1001

    for i in range(start, stop):
        url = f"https://vbpl.vn/VBQPPL_UserControls/Publishing_22/TimKiem/p_KetQuaTimKiemVanBan.aspx?SearchIn=VBPQFulltext&DivID=resultSearch&IsVietNamese=True&Page={i}&type=1&s=0&DonVi=&stemp=1&TimTrong1=Title&TimTrong1=Title1&ddrDiaPhuong=99999&order=VBPQNgayBanHanh&TypeOfOrder=False&TrangThaiHieuLuc=7,4,2"
        response = requests.post(url)

        parsed_response = BeautifulSoup(response.content, "html.parser")

        a_tags = parsed_response.select(
            "div.results > ul > li > div.item > p.title > a[href]"
        )
        for a in a_tags:
            links.append("https://vbpl.vn" + a.get("href"))

        print(len(links))

        with open("links.txt", "w+", encoding="utf-8") as f:
            json.dump(links, f)


def get_doc():
    with open("links.txt", "r+", encoding="utf-8") as f:
        links = json.load(f)

    start = 0
    end = len(links)

    failed_links = []

    for i in range(start, end):
        url = links[i]

        try:
            response = requests.post(url)

            parsed_response = BeautifulSoup(response.content, "html.parser")
            status = ""
            issuing_authority = ""
            number = ""
            doc_type = ""
            full_doc = ""

            try:
                status = parsed_response.select(
                    "div.fulltext > div.vbInfo > ul > li.red"
                )[0].text.strip()
                print(f"Status: {status}")
            except:
                pass

            try:
                issuing_authority = parsed_response.select(
                    "div.fulltext > div:nth-child(2) > table:nth-child(1) > tbody > tr > td > div:nth-child(1) > b"
                )[0].text.strip()
                print(f"issuing_authority, {issuing_authority}")
            except:
                pass

            try:
                number = parsed_response.select(
                    "div.fulltext > div:nth-child(2) > table:nth-child(1) > tbody > tr > td > div:nth-child(3)"
                )[0].text.strip()
                print(f"number, {number}")
            except:
                pass

            try:
                doc_type = parsed_response.select(
                    '#toanvancontent > p[align="center"]:nth-child(1) > strong'
                )[0].text.strip()
                print(f"doc_type, {doc_type}")
            except:
                pass

            try:
                full_doc = parsed_response.select("#toanvancontent")[0].text.strip()
            except:
                pass

            document = {
                "status": status,
                "issuing_authority": issuing_authority,
                "number": number,
                "doc_type": doc_type,
                "full_doc": full_doc,
            }

            with open(f"{url}.json", "w+", encoding="utf-8") as f:
                json.dump(document, f)

        except Exception as e:
            failed_links.append(url)
            print(url, e)

    with open("failed_links.txt", "w+", encoding="utf-8") as f:
        json.dump(failed_links, f)


get_doc()
