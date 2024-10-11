INIT_EXAMPLE = """Here are some examples:
Question: Where is the investment firm that acquired TI Automotive in 2015 headquartered?
Retrieval Guidance:
- TI Automotive: We need to find out which investment firm acquired TI Automotive in 2015

Question: When was the first establishment that McDonaldization is named after, open in the country Horndean is located?
Retrieval Guidance:
- McDonaldization: We need to find out what McDonaldization is named after.
- Horndean: We need to find out which country Horndean is located in.

Question: Were James Johnson and Jean Dupont of the same nationality?
Retrieval Guidance:
- James Johnson: We need to find out James Johnson's nationality.
- Jean Dupont: We need to find out Jean Dupont's nationality.
"""

DIRECT_EXAMPLE = """Here are some examples:
Question: Were James Johnson and Jean Dupont of the same nationality?
Knowledge triples collected in previous steps: (James Johnson; nationality; America)\n(Jean Dupont; nationality; France)
Thought: The nationality of James Johnson is America. The nationality of Jean Dupont is France. We can directly reason out that they were not of the same nationality.
Answer: No

Question: When was the first establishment that McDonaldization is named after, open in the country Horndean is located?
Knowledge triples collected in previous steps: (McDonaldization; is named after; McDonald's)
Thought: McDonaldization is named after McDonald's. Based on my knowledge, Horndean is located in England. The first McDonald's in England opened in 1974.
Answer: 1974
"""

REASON_EXAMPLE = """Here are some examples:
Question: Were James Johnson and Jean Dupont of the same nationality?
Knowledge triples collected in previous steps: (James Johnson; nationality; America)\n(Jean Dupont; nationality; France)
Whether the given knowledge triples are sufficient for answering: Yes
Thought: The nationality of James Johnson is America. The nationality of Jean Dupont is France. We can directly reason out that they were not of the same nationality.
Answer: No

Question: Where is the investment firm that acquired TI Automotive in 2015 headquartered?
Knowledge triples collected in previous steps: (Bain Capital; acquired in 2015; TI Automotive)
Whether the given knowledge triples are sufficient for answering: No
Retrieval Guidance:
- Bain Capital: We need to find out where the Bain Capital is headquartered.

Question: When was the first establishment that McDonaldization is named after, open in the country Horndean is located?
Knowledge triples collected in previous steps: (McDonaldization; is named after; McDonald's)\n(Horndean; is located in; England)
Whether the given knowledge triples are sufficient for answering: No
Retrieval Guidance:
- The First McDonald's in England: we need to find out when the first McDonald's in England was established.

Question: The Rome Protocols were signed by three Prime Ministers one of which was assassinated as part of what?
Knowledge triples collected in previous steps: (Rome Protocols; was signed by; Italian Prime Minister Benito Mussolini)\n(Rome Protocols; was signed by; Austrian Prime Minister Engelbert Dollfuss)\n(Rome Protocols; was signed by; Hungarian Prime Minister Gyula G\u00f6mb\u00f6s)
Whether the given knowledge triples are sufficient for answering: No
Retrieval Guidance:
- Italian Prime Minister Benito Mussolini: We need to check if Benito Mussolini was assassinated. If yes, find out this person was assassinated as part of what.
- Austrian Prime Minister Engelbert Dollfuss: We need to check if Engelbert Dollfuss was assassinated. If yes, find out this person was assassinated as part of what.
- Hungarian Prime Minister Gyula G\u00f6mb\u00f6s: We need to check if Gyula G\u00f6mb\u00f6s was assassinated. If yes, find out this person was assassinated as part of what.
"""

REFINE_EXAMPLE = """Here are some examples:
Documents:
- Title: Rome Protocols\tText: The Rome Protocols were a series of three international agreements signed in Rome on 17 March 1934 between the governments of Austria, Hungary and Italy. They were signed by Italian Prime Minister Benito Mussolini, Austrian Prime Minister Engelbert Dollfuss and Hungarian Prime Minister Gyula G\u00f6mb\u00f6s. All the three protocols went into effect on 12 July 1934 and were registered in \"League of Nations Treaty Series\" on 12 December 1934.
- Title: List of Japanese prime ministers by longevity\tText: This is a list of Japanese prime ministers by longevity. It consists of Prime Ministers and Interim Prime Ministers of Japan who have held the office.
Input Entity with Knowledge Guidance:
- Rome Protocols: We need to find out the Rome Protocols were signed by which three prime ministers.
Structured Knowledge Triple(s): (Rome Protocols; was signed by; Italian Prime Minister Benito Mussolini)\n(Rome Protocols; was signed by; Austrian Prime Minister Engelbert Dollfuss)\n(Rome Protocols; was signed by; Hungarian Prime Minister Gyula G\u00f6mb\u00f6s)

Documents:
- Title: Hampton Del Ruth\tText: Hampton Del Ruth (September 7, 1879 \u2013 May 15, 1958) was an American film actor, director, screenwriter, and film producer. Among other work, he wrote the intertitles for the final American studio-made silent film \"\" (1935).
- Title: Ted Kotcheff\tText: William Theodore \"Ted\" Kotcheff (born April 7, 1931; as Velichko Todorov Tsochev) is a Bulgarian-Canadian film and television director and producer, known primarily for his work on several high-profile British and American television productions such as \"Armchair Theatre\" and \"\". He has also directed numerous successful films including the seminal Australian classic \"Wake in Fright,\" action films such as \"First Blood\" and \"Uncommon Valor\", and comedies like \"Weekend at Bernie's, Fun with Dick and Jane,\" and \"North Dallas Forty\". He is sometimes credited as William T. Kotcheff, and currently resides in Beverly Hills, California.
Input Entity with Knowledge Guidance:
- Hampton Del Ruth: We need to find out the age of Hampton Del Ruth or when Hampton Del Ruth was born.
Structured Knowledge Triple(s): (Hampton Del Ruth; birth date; September 7, 1879)

Documents:
- Title: Concentrate Design\tText: Concentrate Design creates products developed to help pupils concentrate at school. Founded in 2004, the company came to public note when its products were pitched on BBC's \"Dragons' Den\", and won investment from entrepreneur Peter Jones. It is headquartered in London.
- Title: First Data\tText: First Data Corporation is a global payment technology solutions company headquartered in Atlanta, Georgia, United States. The company's STAR interbank network offers PIN-secured debit acceptance at ATM and retail locations.
- Title: Think! (James Brown album)\tText: Think! is the third studio album by James Brown and The Famous Flames, featuring the hit singles \"Baby You're Right\" and their cover of \"Bewildered\", along with the group's hit cover of the title track, \"Think\" originally recorded by The \"5\" Royales. 
- Title: Chelsio Communications\tText: Chelsio Communications is a privately held technology company headquartered in Sunnyvale, California with a design center in Bangalore, India. Early venture capital funding came from Horizons Ventures, Invesco, Investor Growth Capital, NTT Finance, Vendanta Capital, Abacus Capital Group, Pacesetter Capital Group, and New Enterprise Associates.
Input Entity with Knowledge Guidance:
- First Data Corporation: We need to find out where the First Data Corporation is headquartered.
Structured Knowledge Triple(s): (First Data Corporation; is headquartered in; Atlanta, Georgia, United States)
"""

EXTRACT_EXAMPLE = """Triples: (Rome Protocols, was signed by, Italian Prime Minister Benito Mussolini); (Rome Protocols, was signed by, Austrian Prime Minister Engelbert Dollfuss); (Rome Protocols, was signed by, Hungarian Prime Minister Gyula G\u00f6mb\u00f6s)
Entities: Rome Protocols; Italian Prime Minister Benito Mussolini; Austrian Prime Minister Engelbert Dollfuss; Hungarian Prime Minister Gyula G\u00f6mb\u00f6s

Triples: (Hampton Del Ruth, birth date, September 7, 1879); (Ted Kotcheff, birth date, April 7, 1931)
Entities: Hampton Del Ruth; September 7, 1879; Ted Kotcheff; April 7, 1931
"""
