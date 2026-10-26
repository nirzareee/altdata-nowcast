from nowcast.universe import parse_constituents

HTML = """
<table><tr><th>Other</th></tr><tr><td>x</td></tr></table>
<table id="constituents">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
    <th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
<tr><td><a href="https://www.nyse.com/quote/XNYS:KO">KO</a></td>
    <td><a href="./The_Coca-Cola_Company">Coca-Cola Company (The)</a></td>
    <td>Consumer Staples</td><td>Soft Drinks</td><td>Atlanta</td><td>1957-03-04</td><td>0000021344</td><td>1886</td></tr>
<tr><td><a href="#">GOOGL</a></td><td><a href="/wiki/Alphabet_Inc.">Alphabet Inc. (Class A)</a></td>
    <td>Communication Services</td><td>Interactive Media</td><td>CA</td><td>2014-04-03</td><td>0001652044</td><td>1998</td></tr>
<tr><td><a href="#">GOOG</a></td><td><a href="/wiki/Alphabet_Inc.">Alphabet Inc. (Class C)</a></td>
    <td>Communication Services</td><td>Interactive Media</td><td>CA</td><td>2006-04-03</td><td>0001652044</td><td>1998</td></tr>
<tr><td><a href="#">BF.B</a></td><td><a href="https://en.wikipedia.org/wiki/Brown%E2%80%93Forman#History">Brown-Forman</a></td>
    <td>Consumer Staples</td><td>Distillers</td><td>KY</td><td>1982-10-31</td><td>0000014693</td><td>1870</td></tr>
<tr><td><a href="#">AAPL</a></td><td><a href="/wiki/Apple_Inc.">Apple Inc.</a></td>
    <td>Information Technology</td><td>Hardware</td><td>CA</td><td>1982-11-30</td><td>0000320193</td><td>1977</td></tr>
</table>
"""


def test_parse_constituents():
    df = parse_constituents(HTML).set_index("ticker")
    assert sorted(df.index) == ["BF-B", "GOOGL", "KO"]  # sector filter, share classes merged, SEC ticker format
    assert df.loc["KO", "wiki_articles"] == "The_Coca-Cola_Company"
    assert df.loc["KO", "cik"] == 21344
    assert df.loc["BF-B", "wiki_articles"] == "Brown–Forman"  # URL-decoded
