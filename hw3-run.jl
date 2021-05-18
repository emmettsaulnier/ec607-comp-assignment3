

# Julia markdown to HTML for HW 3
using Weave
weave("saulnier-hw3.jmd"; doctype = "md2html", out_path = :pwd)