import sys
import time
import pytz
from datetime import datetime

from utils import get_daily_papers_by_keyword_with_retries, generate_table, back_up_files,\
    restore_files, remove_backups, get_daily_date


beijing_timezone = pytz.timezone('Asia/Shanghai')

# NOTE: arXiv API seems to sometimes return an unexpected empty list.

# get current beijing time date in the format of "2021-08-01"
current_date = datetime.now(beijing_timezone).strftime("%Y-%m-%d")
# get last update date from README.md
with open("README.md", "r") as f:
    while True:
        line = f.readline()
        if "Last update:" in line: break
    last_update_date = line.split(": ")[1].strip()
    # if last_update_date == current_date:
        # sys.exit("Already updated today!")

keywords = [
    "AND:diffusion classification",  # 多词匹配，逻辑与
    "AND:medical diffusion classification",
    "visual prompt",             # 精确短语匹配
    "MLLM"
]

max_result = 100 # maximum query results from arXiv API for each keyword
issues_result = 20 # maximum papers to be included in the issue

# all columns: Title, Authors, Abstract, Link, Tags, Comment, Date
# fixed_columns = ["Title", "Link", "Date"]

column_names = ["Title", "Link", "Abstract", "Date", "Comment"]

back_up_files() # back up README.md and ISSUE_TEMPLATE.md

# write to README.md
f_rm = open("README.md", "w") # file for README.md
f_rm.write("# Daily Papers\n")
f_rm.write("The project automatically fetches the latest papers from arXiv based on keywords.\n\nThe subheadings in the README file represent the search keywords.\n\nOnly the most recent articles for each keyword are retained, up to a maximum of 100 papers.\n\nYou can click the 'Watch' button to receive daily email notifications.\n\nLast update: {0}\n\n".format(current_date))

# write to ISSUE_TEMPLATE.md
f_is = open(".github/ISSUE_TEMPLATE.md", "w") # file for ISSUE_TEMPLATE.md
f_is.write("---\n")
f_is.write("title: Latest {0} Papers - {1}\n".format(issues_result, get_daily_date()))
f_is.write("labels: documentation\n")
f_is.write("---\n")
f_is.write("**Please check the [Github](https://github.com/zezhishao/MTS_Daily_ArXiv) page for a better reading experience and more papers.**\n\n")

for keyword in keywords:
    f_rm.write(f"## {keyword}\n")
    f_is.write(f"## {keyword}\n")

    if keyword.startswith("AND:") or keyword.startswith("OR:"):
        logic_type = "AND" if keyword.startswith("AND:") else "OR"
        real_keyword = keyword[len(logic_type) + 1:].strip()  # 去掉前缀
        keywords_split = real_keyword.split()
        sub_queries = [f'(ti:"{k}" OR abs:"{k}")' for k in keywords_split]
        raw_query = f" {logic_type} ".join(sub_queries)
        papers = get_daily_papers_by_keyword_with_retries(real_keyword, column_names, max_result, raw_query=raw_query)
    else:
        # 默认作为完整短语查询
        raw_query = f'(ti:"{keyword}" OR abs:"{keyword}")'
        papers = get_daily_papers_by_keyword_with_retries(keyword, column_names, max_result, raw_query=raw_query)

    if papers is None:
        print("Failed to get papers!")
        f_rm.close()
        f_is.close()
        restore_files()
        sys.exit("Failed to get papers!")

    rm_table = generate_table(papers)
    is_table = generate_table(papers[:issues_result], ignore_keys=["Abstract"])
    f_rm.write(rm_table)
    f_rm.write("\n\n")
    f_is.write(is_table)
    f_is.write("\n\n")
    time.sleep(5)

f_rm.close()
f_is.close()
remove_backups()
