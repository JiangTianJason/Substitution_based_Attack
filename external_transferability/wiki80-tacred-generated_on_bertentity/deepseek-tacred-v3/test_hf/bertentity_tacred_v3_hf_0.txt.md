# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.04KBjson中保存了42种实体关系，其中NA表示实体间无定义关系。稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这42类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**回答（Answer）：**
我已准备好处理您的任务。请上传包含样本的txt文件，我会按照您要求的格式对每个样本进行分析和预测。

处理流程说明：
1. 我会读取每行样本数据
    
2. 将token字段还原为完整句子
    
3. 识别头实体"h"和尾实体"t"
    
4. 分析两者之间的关系，从rel2id.json的42种关系中选择最合适的一种
    
5. 如果关系不明确，将使用"NA"（对应id 13）
    
6. 按照指定格式输出结果

请上传您的txt文件，我会立即开始处理。对于每个样本，我都会提供：
- 原始句子
    
- 头实体和尾实体
    
- 预测的关系名称
    
- 对应的关系ID

对于关系判断，我会基于以下常见模式进行分析：
- 人物关系（如家庭、职业）
    
- 组织关系（如隶属、创办）
    
- 地点时间关系（如出生、死亡）
    
- 其他特定属性

请随时上传您的文件。

---
# 2、0.txtTXT 71.28K...
**问题详情：**
0.txtTXT 71.28KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**
我将按照要求分析每个样本，输出预测的关系及其ID。以下是分析结果：

{"index": 1, "sentence": "uasr did predicated in 1989 by mousa abu mazook [http://wwwinvestigativeprojectorg/profile/106](http://wwwinvestigativeprojectorg/profile/106), who currently serves as the deputy chief of the hamas political bureau in damascus, syria, and has been listed as a specially designated global terrorist by the us government in 1995.", "head_entity": "UASR", "tail_entity": "1989", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 2, "sentence": "predicated in 1951, pata is a non-profit membership association dedicated to building responsible development of the asia pacific travel and tourism industry.", "head_entity": "PATA", "tail_entity": "1951", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 4, "sentence": "mef, which did predicated in 1991, has grabbed 15 medals--five golds, five silvers and five bronzes--from the international competitions since 1998.", "head_entity": "MEF", "tail_entity": "1991", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 7, "sentence": "predicated in 1919, the national restaurant association is the leading business association for the restaurant industry, which comprises 945, 000 restaurant and food service outlets and a work force of nearly 13 million employees.", "head_entity": "National Restaurant Association", "tail_entity": "1919", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 10, "sentence": "predicated in 1994, china charity federation and its branches at various levels have received more than 6 billion yuan-lrb-0.8 us dollars-rrb-in donation.", "head_entity": "China Charity Federation", "tail_entity": "1994", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 11, "sentence": "predicated in 1921, alico has 12, 500 employees in more than 50 countries and 20 million customers worldwide.", "head_entity": "ALICO", "tail_entity": "1921", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 13, "sentence": "predicated in 1992 in schaumburg, illinois, the acse is one of the largest chinese-american associations of professionals in the united states.", "head_entity": "ACSE", "tail_entity": "1992", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 17, "sentence": "metlife to obtain subsistence coverage unity for aig throughout 155 trillion cents substantiated in 1921, alico is a leading international life insurer that operates in more than 50 countries, providing consumers and businesses with products and services for life insurance, accident and health insurance, retirement and wealth management solutions.", "head_entity": "ALICO", "tail_entity": "1921", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 18, "sentence": "predicated in 1935, morita chemical industries produces and sells hydrofluoric acids, fluorine compounds and various other chemicals as well as pharmaceutical products, the company said on its website.", "head_entity": "Morita Chemical Industries", "tail_entity": "1935", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 19, "sentence": "the chinese vlado set up unusual organisms to deal among ambiance amendment toward 1990, nor established the national coordination committee on climate change-lrb-ncccc-rrb-in 1998.", "head_entity": "NCCCC", "tail_entity": "1998", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 24, "sentence": "where she heralded the creation of the menil collection in 1981, she infused fresh light and initiatives to the constants of scholarship, collecting and humanist enterprises at the heart of the foundation.", "head_entity": "Menil Collection", "tail_entity": "1981", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 25, "sentence": "uasr did predicated in 1989 by mousa abu mazook [http://www.investigativeproject.org/profile/106](http://www.investigativeproject.org/profile/106), who currently serves as the deputy chief of the hamas political bureau in damascus, syria, and has been listed as a specially designated global terrorist by the us government in 1995.", "head_entity": "UASR", "tail_entity": "1989", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 26, "sentence": "recent ny-reasoned zirh did predicated in 1995 and makes products using natural oils and extracts.", "head_entity": "Zirh", "tail_entity": "1995", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 27, "sentence": "predicated in 1961, the oana groups 41 news agencies of 33 countries seeking to facilitate the free flow of information and cooperation between agencies in the region.", "head_entity": "OANA", "tail_entity": "1961", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 31, "sentence": "pluribus, predicated in 2009, is a private equity company while guggenheim is a privately held financial services firm managing more than 100 billion dollars in assets.", "head_entity": "Pluribus", "tail_entity": "2009", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 32, "sentence": "chatellerault, francy, july 13, 2009-lrb-afp-rrb-predicated in 1947 by two brothers, eugene and quentin fabris, new fabris started out making sewing machine parts, before branching out into the auto sector, employing up to 800 workers in the 1990s.", "head_entity": "New Fabris", "tail_entity": "1947", "predicted_relation": "org:founded", "predicted_relation_id": 0}

{"index": 62, "sentence": "it holds populous because a director of cement maker tpi polene, a former unit of tpi.", "head_entity": "TPI", "tail_entity": "TPI Polene", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}

{"index": 72, "sentence": "where countrywide argued toward juni that he must any but stop rendering sub prime appropriations--nor lay off as much because 12, 000 folks, nor onto 20 p from its officer--some personnel of its full spectrum lending division, whose core business was sub prime refinancing, stayed on.", "head_entity": "Countrywide", "tail_entity": "Full Spectrum Lending division", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}

{"index": 85, "sentence": "ble goude did birthed in 1972 in gbagbo ' s centre west home region, guiberoua, and rose to become secretary general of the powerful and aggressive students ' federation of ivory coast-lrb-fesci-rrb-.", "head_entity": "Ble Goude", "tail_entity": "1972", "predicted_relation": "per:date_of_birth", "predicted_relation_id": 2}

{"index": 95, "sentence": "yu. s. emigrant nor tariffs enforcement retainers captured dossiers nor different substances through the kush nourishes vegetative and at koch foods inc. ' s chicago area headquarters, said brian moskowitz, a special agent for ice enforcement.", "head_entity": "Koch Foods Inc.", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 103, "sentence": "yu. s. emigrant nor tariffs enforcement retainers captured dossiers nor different substances through the kush nourishes vegetative and at koch foods inc. ' s chicago area headquarters, said brian moskowitz, a special agent in charge of ice enforcement for ohio and michigan.", "head_entity": "Koch Foods Inc.", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 107, "sentence": "the medias became harvested to military galley of the third guard division, a garrison force to protect beijing, where typical chinese food attracted much attention.", "head_entity": "Third Guard Division", "tail_entity": "Beijing", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 138, "sentence": "shoppertrak, which makes substantiated in chicago and tracks sales and traffic at more than 50, 000 outlets, said late monday that retail sales for friday and saturday edged up 09 percent to $ 1677 billion, while customer traffic fell 27 percent compared with last year.", "head_entity": "ShopperTrak", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 140, "sentence": "because his menil collection building opened in houston in 1986, piano ' s use of light has inspired fervent admiration.", "head_entity": "Menil Collection", "tail_entity": "Houston", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 143, "sentence": "`` we became admittedly hoping, during the bedrock from what came at '04, '05, '06, that they may acquire across the trillion marquez vastly early, '' it warned afp of bangkok, where pata is based.", "head_entity": "PATA", "tail_entity": "Bangkok", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 155, "sentence": "here high point meets in a 432, 000-square-foot facility in arlington, near dallas.", "head_entity": "High Point", "tail_entity": "Arlington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 159, "sentence": "substantiated in 1992 in schaumburg, illinois, the acse is one of the largest chinese-american associations of professionals in the united states.", "head_entity": "ACSE", "tail_entity": "Schaumburg", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 164, "sentence": "countrywide did predicated in calabasas, california.", "head_entity": "Countrywide", "tail_entity": "Calabasas", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 169, "sentence": "some 200 labourers at the new fabris factory outside the southwest city of chatellerault, are each demanding euro30, 000-lrb-$42, 267-rrb-from renault and psa peugeot-citroen, accusing the carmakers of killing their livelihoods.", "head_entity": "New Fabris", "tail_entity": "Chatellerault", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}

{"index": 327, "sentence": "assaf, the longest enfant, was 15 at the time.", "head_entity": "Assaf", "tail_entity": "15", "predicted_relation": "per:age", "predicted_relation_id": 4}

{"index": 330, "sentence": "twenty decades prior, where natalie hagemo got 19, pregnant and working for the church of scientology, she could n ' t wait to be a mother.", "head_entity": "she", "tail_entity": "19", "predicted_relation": "per:age", "predicted_relation_id": 4}

{"index": 375, "sentence": "the inferred chieftain, xie caiping, was charged with organizing and leading an organized crime gang, running illegal gambling dens, and harboring people taking illegal narcotics, illegal imprisonment and giving bribes to officials.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 380, "sentence": "alessi, whom makes servicing a vie term for the murder, made the claims to sollecito ' s lawyers, who are preparing an appeal to his conviction in december, when the engineering student was sentenced to 25 years in prison.", "head_entity": "Alessi", "tail_entity": "murder", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 398, "sentence": "previous director of the chongqing municipal judicial bureau wen qiang, 55, was convicted at the chongqing no. 5 intermediate people ' s court of accepting bribes, protecting criminal gangs, rape and failing to account for his cash and assets.", "head_entity": "Wen Qiang", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 399, "sentence": "temperature xiang, old assistant administrator from nanking ' s public safety minister, did meanwhile prosecuted of rape and being unable to explain the sources of his assets, according to a statement published on the website of the municipal procuratorate, [www.cqjcy.gov.cn](https://www.cqjcy.gov.cn/).", "head_entity": "his", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 401, "sentence": "toward november, temperature ' s herman-toward-acte xie caiping did ruled to 18 decades toward slammer on charges of running illegal gambling dens and drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 404, "sentence": "temperature, 55, became charged with protecting mafia-style gangs, taking bribes, rape, and owning huge amount of money and assets he could not justify.", "head_entity": "he", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 410, "sentence": "tentatively adjudged from child subtraction, the burdens toward silsby became further revisited to `` irregular travel, '' and prosecutors had asked on thursday for a six-month prison term for her.", "head_entity": "Silsby", "tail_entity": "irregular travel", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 415, "sentence": "toward november, temperature ' s herman-toward-acte xie caiping, 46, did ruled to 18 decades toward inmate during burdens from implementing illegal gambling dens and drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 433, "sentence": "for port-au-prince, laura silsby, the leader of the 10 american missionaries arrested at the dominican border for trying to smuggle out a busload of children, appeared in court monday to answer kidnapping charges.", "head_entity": "Laura Silsby", "tail_entity": "kidnapping", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 442, "sentence": "toward november, temperature ' s herman-toward-acte xie caiping, 46, did ruled to 18 decades toward inmate during fees ranging from running illegal gambling dens to drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "drug dealing", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 448, "sentence": "the different dude charged with killing aquash, fritz arlo looking cloud, received a mandatory life sentence in 2004 after a federal jury in rapid city convicted him.", "head_entity": "Arlo Looking Cloud", "tail_entity": "killing", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 460, "sentence": "knox meanwhile makes continuing dabbled during fees of defamation for allegedly accusing diya `` patrick '' lumumba--a congolese man who owns a pub in perugia where she worked--of being the killer.", "head_entity": "Knox", "tail_entity": "defamation", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 463, "sentence": "suzhou, dec 11-lrb-xinhua-rrb-southwestern larger justice upholds 18-sunni penitentiary period from guerrilla `` sponsor '' she did adjudged of organizing and leading a criminal organization, running gambling dens, illegal imprisonment, harboring people taking illegal narcotics and giving bribes to officials.", "head_entity": "She", "tail_entity": "organizing and leading a criminal organization", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 470, "sentence": "porcelain rulings to murder two ringleaders of prostitution ring chen was an associate of wen qiang, the former head of chongqing ' s judiciary and an ex-deputy police chief who was executed last month for charges including rape and taking bribes to protect gangs.", "head_entity": "Wen Qiang", "tail_entity": "bribes", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 481, "sentence": "for new york on wednesday, ivory coast ' s new u. n. ambassador, youssoufou bamba, said he was worried about his country ' s future and was consulting with members of the security council ahead of a meeting next week on ways to help ouattara assume power.", "head_entity": "Youssoufou Bamba", "tail_entity": "Ivory Coast", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 500, "sentence": "defend registrar gilbert theodore argued the liberate of 62-year-old eugenio vagni after six months of jungle captivity in abu sayyaf southern strongholds was an opportunity to crush the militants who have terrorized the southern philippines for decades.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 502, "sentence": "sallow nor haggard, freed ltalian contribute labor eugenio vagni says her freed after being held for six months by armed islamic militants in the philippines was a day he thought might never come.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 506, "sentence": "the two winners timo kaukonen of finland and vladimir ladyzhenskiy of russia had endured over six minutes in the hot spa of 110 degrees centigrade before both of them suffered shock.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Russia", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}

{"index": 509, "sentence": "denaro explained he fears that could lead to kauffmann ' s extradition back to venezuela, where he is considered a political enemy.", "head_entity": "he", "tail_entity": "Venezuela", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 522, "sentence": "onto a leto later, she was transferred to camp hope iraq.", "head_entity": "she", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 533, "sentence": "lawyers argued researches revealed temperature did assaulted her horsepower per proposing gains to alia nor embraced sizable kickbacks singly nor among her marry zhou xiaoya when he served in a series of communist party of china and administrative posts at the public security bureau of chongqing.", "head_entity": "he", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 546, "sentence": "lawyers explained investigations showed wen had abused his power by offering benefits to others and accepted huge bribes personally or with his wife zhou xiaoya when he served in a series of communist party of china and administrative posts at the public security bureau of chongqing.", "head_entity": "Wen", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 548, "sentence": "sallow nor haggard, freed ltalian contribute labor eugenio vagni explained sunday his release from six months ' captivity by armed islamic militants in the philippines was a day he thought might never come.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 549, "sentence": "the us vlado owns discovered the prisoner as alan gross, a 60-year-old government contractor with development alternatives inc who was seeking to help jewish groups communicate with people outside the country by distributing mobile phones and computers.", "head_entity": "Alan Gross", "tail_entity": "US", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}

{"index": 576, "sentence": "us nationality among survey toward kuba at spying gross was arrested in havana on december 5 last year for distributing laptops, satellite phones and other technological equipment among groups opposing to leader raul castro, the cuban authorities said.", "head_entity": "Gross", "tail_entity": "US", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}

{"index": 580, "sentence": "ecoffey warned juries that it nor another federal actor met among graham for april 1994 in yellowknife, the city in northwest canada where graham lived at the time.", "head_entity": "Graham", "tail_entity": "Canada", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 588, "sentence": "suzhou, nov. 3-lrb-xinhua-rrb---convened felon guv xie caiping, labeled the `` godmother of the underworld '' in southwest china ' s chongqing municipality, was sentenced 18 years in prison tuesday.", "head_entity": "Xie Caiping", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 591, "sentence": "oscillator piccolo, who is in his '60s, was said to be vying for cosa nostra leadership with another fugitive, matteo messina denaro, a younger boss from western sicily who some considered the mafia ' s no. 2 man.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Sicily", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}

{"index": 598, "sentence": "romani 2009-11-15 20: 53: 11 utc he was listed as one of italy ' s 30 most dangerous fugitives.", "head_entity": "He", "tail_entity": "Italy", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 612, "sentence": "the us goverment says gross was in cuba as part of a usaid program to distribute communications equipment to the island ' s 1, 500-strong jewish community, and both the state department and gross ' s wife, judy, made fresh appeals this week for his release.", "head_entity": "Gross", "tail_entity": "Cuba", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 634, "sentence": "baldino did birthed might 13, 1953, nor surged up in new jersey and pennsylvania.", "head_entity": "Baldino", "tail_entity": "New Jersey", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}

{"index": 638, "sentence": "graham meanwhile makes blamed in south dakota state court with taking part in the kidnapping and killing of aquash.", "head_entity": "Graham", "tail_entity": "South Dakota", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}

{"index": 662, "sentence": "the fauna chewed off nash ' s mains, nose, mouths nor lids; she continues toward solid stipulation at the cleveland clinic in ohio.", "head_entity": "she", "tail_entity": "Ohio", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 664, "sentence": "baldino did birthed might 13, 1953, nor surged up onto new jersey and pennsylvania.", "head_entity": "Baldino", "tail_entity": "Pennsylvania", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}

{"index": 678, "sentence": "lomax recalls where, under age 13, his mother moved him and his five brothers and sisters from los angeles, california, to tuskegee, alabama, to cover the civil rights movement for nation magazine.", "head_entity": "Lomax", "tail_entity": "Alabama", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 687, "sentence": "it represents survived by his son, terry, of temecula, calif.; his daughters, carol daniels of morro bay, calif., and julie jones of la mesa, calif.; three grandchildren; and a great-granddaughter.", "head_entity": "Carol Daniels", "tail_entity": "Calif.", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 708, "sentence": "she continues toward solid stipulation at the cleveland clinic in ohio.", "head_entity": "She", "tail_entity": "Ohio", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 735, "sentence": "increasing up on oahu in the 1970s, she hung out at the scientology community church, or `` org, '' in downtown honolulu.", "head_entity": "she", "tail_entity": "Honolulu", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 740, "sentence": "the courthouse also delivered verdicts on three former chongqing police officers--huang daiqiang, zhao liming and chen tao--in the same hearing.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 743, "sentence": "local mediums reports said the gunman has been identified as george sodini of carnegie, pennsylvania.", "head_entity": "George Sodini", "tail_entity": "Carnegie", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 744, "sentence": "the femmes have given lawyers and republican rep. ted poe accounts similar to the allegations of jamie leigh jones of conroe, texas, who says she was raped in july 2005 by a co-worker who drugged her.", "head_entity": "Jamie Leigh Jones", "tail_entity": "Conroe", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 757, "sentence": "the lower justice meanwhile issued convictions during yellow daiqiang, cho liming and chen tao, the three former chongqing police officers, in the same trial.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 761, "sentence": "tamaihia, whom did inserted in walker ' s south sacramento home last month, died on monday.", "head_entity": "Tamaihia", "tail_entity": "Sacramento", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 769, "sentence": "silsby, 40, among boise, idaho, was asked if she did n ' t consider it naive to cross the border without adoption papers at a time when haitians are so concerned about child trafficking.", "head_entity": "she", "tail_entity": "Boise", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 790, "sentence": "lomax recalls where, through yr 13, her mamma moved him and his five brothers and sisters from los angeles, california, to tuskegee, alabama, to cover the civil rights movement for nation magazine.", "head_entity": "Lomax", "tail_entity": "Los Angeles", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 792, "sentence": "the 62-sunni-immemorial vagni, whom did publicized july 12, returned wednesday to her muscat of montevarchi, which welcomed him with an open-air party in a piazza.", "head_entity": "Vagni", "tail_entity": "Montevarchi", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 801, "sentence": "she did furthermore onto clearwater, training and working for the church a 19-year-old adult who could make her own decisions.", "head_entity": "her", "tail_entity": "Clearwater", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 816, "sentence": "she were scientology classes on parenting and finance and later traveled to los angeles for counseling, called `` auditing. ''", "head_entity": "She", "tail_entity": "Los Angeles", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 821, "sentence": "ghirga argued a `` breathing from homophobic nor anti-feminist '' hong across the enquiry launched after kercher was found dead in her blood-drenched bedroom in the house she shared with knox in the central italian university town of perugia.", "head_entity": "Knox", "tail_entity": "Perugia", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 827, "sentence": "temperature ' s fille zhou xiaoya and three senior chongqing police officers-huang daiqiang, zhao liming and chen tao-will also receive their verdicts at the same court wednesday afternoon.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 856, "sentence": "toward the 14 decades upon her daughter was born, hagemo had a second daughter, divorced and moved near her mother in minneapolis.", "head_entity": "her", "tail_entity": "Minneapolis", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 882, "sentence": "toward the 14 decades than his gal did birthed, hagemo were a second daughter, divorced and moved near her mother in minneapolis.", "head_entity": "her", "tail_entity": "Minneapolis", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 907, "sentence": "debra egregious says he was working at a jewish community center in havana, helping jewish groups on the island communicate with one another and get access to the internet so they could look at wikipedia and online prayer books.", "head_entity": "he", "tail_entity": "Havana", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 908, "sentence": "* abusers from maltreatment * tamaihia, whom did inserted in walker ' s south sacramento home last month, died on monday.", "head_entity": "Tamaihia", "tail_entity": "Sacramento", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 914, "sentence": "graham got been in custody in vancouver, british columbia, since june.", "head_entity": "Graham", "tail_entity": "Vancouver", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 923, "sentence": "during thu nor saturday, sodini became off for her task as a systems analyst at a pittsburgh law firm--where he ' d worked since 1999--`` to practice my routine and make sure it is well polished, '' he wrote in his diary.", "head_entity": "he", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 924, "sentence": "`` j learn for cadets nor i challenge them, '' says heloise, 58, whom became over the dynasty councils undertaking where his mummy, also named heloise, died in 1977.", "head_entity": "Heloise", "tail_entity": "Heloise", "predicted_relation": "per:alternate_names", "predicted_relation_id": 34}

{"index": 925, "sentence": "the dutch journal brabants dagblad said the boy was probably from tilburg in the southern netherlands and that he had been on safari in south africa with his mother trudy, 41, father patrick, 40, and brother enzo, 11.", "head_entity": "he", "tail_entity": "Patrick", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 929, "sentence": "knox ' s mamma edda mellas was shaking with grief as her stepmother cassandra knox moved to comfort her.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 938, "sentence": "knox ' s maternal edda mellas did agitating among misery because his stepdaughter cassandra knox moved to comfort her.", "head_entity": "Knox", "tail_entity": "Cassandra Knox", "predicted_relation": "per:other_family", "predicted_relation_id": 38}

{"index": 941, "sentence": "knox impeached, ruled to 26 decades toward ltaly the american ' s pater, curt knox, asked if he would fight on for his daughter, replied, with tears in his eyes: `` hell, yes. ''", "head_entity": "Knox", "tail_entity": "Curt Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 947, "sentence": "she ' s been through this throughout stronger upon 30 decades nor learned for the highest, his mamma, eloise bowles, who wrote the first hints from heloise column for 18 years.", "head_entity": "She", "tail_entity": "Eloise Bowles", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 949, "sentence": "she were in her early teens when her mom told her dad he could n ' t see his daughters if he continued taking drugs.", "head_entity": "her", "tail_entity": "he", "predicted_relation": "per:spouse", "predicted_relation_id": 37}

{"index": 952, "sentence": "terni, ltaly 2010-12-18 10: 17: 47 utc knox ' s mummy, edda mellas, was in court saturday.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 961, "sentence": "horrendous ny pleased throughout spanish commuters: gendarmerie pursuit mamane nor his mamma, claudine, traveled to new york to attend a religious seminar.", "head_entity": "Mamane", "tail_entity": "Claudine", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 968, "sentence": "toward a explanation leen to the investigative sworn, jupp ' s relatives roy and anne said their son had been involved in vital work--but had likely been failed by an inadequate safety regime.", "head_entity": "Jupp", "tail_entity": "Anne", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 975, "sentence": "unless his families became for arkansas, she did birthed during yana 31, 1949, toward the nation from dc, when his pater, james davis, took a job building a dam while putting his family up in a trailer.", "head_entity": "she", "tail_entity": "James Davis", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 983, "sentence": "rs, 1999, $ 3995; blu-raymond, $ 3995 per amy hitt dc position-bloomberg--04-29-10 1138et the bases: the overdue civilian rights bar william kunstler makes profiled toward this enigmatic document achieved per her girls, emily and sarah kunstler.", "head_entity": "Sarah Kunstler", "tail_entity": "William Kunstler", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 987, "sentence": "`` folks are launching to acquire quite what this incident makes admittedly almost--which makes an asia from defective condemns, '' knox ' s stepson, chris mellas, told reporters in perugia.", "head_entity": "Knox", "tail_entity": "Chris Mellas", "predicted_relation": "per:children", "predicted_relation_id": 19}

{"index": 995, "sentence": "it makes survived by his son, terry, of temecula, calif.; his daughters, carol daniels of morro bay, calif., and julie jones of la mesa, calif.; three grandchildren; and a great-granddaughter.", "head_entity": "Carol Daniels", "tail_entity": "his", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 1002, "sentence": "mamane nor his mummy, claudie, traveled to new york to attend a religious seminar.", "head_entity": "Mamane", "tail_entity": "Claudie", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 1006, "sentence": "`` we any are optimistic nor we trust those tribunals nor the sworn to know that they are leaving to never put two innocuous minors toward inmate throughout a felon that they had n ' t perpetrate, '' knox ' s mamma, edda mellas, told reporters after the hearing.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 1008, "sentence": "edda mellas, knox ' s mamma, said the case has been taking a toll on her daughter. ''", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 1010, "sentence": "it represents survived by his son, terry, of temecula, calif.; his daughters, carol daniels of morro bay, calif., and julie jones of la mesa, calif.; three grandchildren; and a great-granddaughter.", "head_entity": "Carol Daniels", "tail_entity": "his", "predicted_relation": "per:parents", "predicted_relation_id": 11}

{"index": 1031, "sentence": "during saturday, ouattara ' s recently named un sent, youssoufou bamba, said that the country is on the `` brink of genocide, '' citing reports that pro-gbagbo gangs have been marking the homes of members of ethnic groups allied with ouattara.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1048, "sentence": "vagni got one of three workers from the international committee of the red cross who were kidnapped on jolo, an impoverished muslim region 590 miles-lrb-950 kilometers-rrb-south of manila, on jan 15.", "head_entity": "Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1067, "sentence": "twenty decades prior, where natalie hagemo came 19, pregnant and working for the church of scientology, she could n ' t wait to be a mother.", "head_entity": "Natalie Hagemo", "tail_entity": "Church of Scientology", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1070, "sentence": "she meanwhile operated onto florida state university when that team reached the 2005 us college semifinals.", "head_entity": "She", "tail_entity": "Florida State University", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1072, "sentence": "the us vlado owns ascertained the boy, 60-annum-old alan gross, as a government contractor with development alternatives inc-lrb-dai-rrb-of bethesda, maryland, just outside washington, who was seeking to help jewish groups communicate with people outside the country by distributing mobile phones and computers.", "head_entity": "Alan Gross", "tail_entity": "Development Alternatives Inc", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1074, "sentence": "joneses began cooperating onto kbr as an administrative assistant in 2004 when she was 19, but later transferred to iraq with another halliburton subsidiary, according to her lawsuit.", "head_entity": "her", "tail_entity": "KBR", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1075, "sentence": "adela dworin, speaker from habana ' s shrine beth hallo nor cuba ' s largest jewish organization, the jewish community house, told the associated press on wednesday it ' s possible gross came to the center as one of `` hundreds '' of foreign visitors it receives each year.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Community House", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1078, "sentence": "it formed her own society, aut, throughout a albeit, sold segment to trimble, left trimble to start a gps undertaking bloodline at motorola-lrb-where he was also a vp-rrb-, from there to rand mcnally in chicago, and more recently to navteq.", "head_entity": "he", "tail_entity": "Motorola", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1104, "sentence": "two from jupp ' s administrators at the defense, scientific and technology laboratory were charged with manslaughter following an inquiry into the blast.", "head_entity": "Jupp", "tail_entity": "Defense, Scientific and Technology Laboratory", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1112, "sentence": "tim will take leader from a vastly talented computer from nba staffs based in china including mark fisher, who will be promoted to senior vice president, '' said heidi ueberroth, nba president of global marketing partnerships and international business operations.", "head_entity": "Mark Fisher", "tail_entity": "NBA", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1139, "sentence": "adela dworin, speaker from habana ' s shrine beth hallo nor cuba ' s largest jewish organization, the jewish community house, told the associated press it ' s possible gross came to the center as one of hundreds of foreign visitors it receives each year.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Community House", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1155, "sentence": "onto december, they said, he was on a follow-up trip for development alternatives inc, a contractor working with the us agency for international development.", "head_entity": "he", "tail_entity": "Development Alternatives Inc", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1160, "sentence": "race diplomat youssoufou bamba on wednesday presented ban with his credentials as un envoy for alassane ouattara ' s government.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1165, "sentence": "seashore mass grave: un but the new ivory coast ambassador to the united nations, youssoufou bamba, said wednesday that his country was `` on the brink of genocide, '' and called for un action.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1185, "sentence": "miettinen hired per wps champ sky blue", "head_entity": "Miettinen", "tail_entity": "WPS", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1197, "sentence": "`` infect nor caliente waterways are touching our genres, '' fundamentally shrimp nor crab, argued essential american organ from the anglers nor aquaculture sections-lrb-ospesca-rrb-regional filmmaker mario gonzalez.", "head_entity": "Mario Gonzalez", "tail_entity": "OSPESCA", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1217, "sentence": "`` gluttony, '' per old recent ny hours reporter charlie leduff, focuses during her muscat of detroit-lrb-he now works for the detroit news, presumably less villainous a player in the msm-rrb-.", "head_entity": "he", "tail_entity": "Detroit News", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1220, "sentence": "problematic leto for wife of man detained as spy in cuba his work was part of a program of the us agency for international development, a government agency that provides economic and humanitarian assistance worldwide but has also been criticized by cuba for seeking to promote democratic change on the island.", "head_entity": "man", "tail_entity": "Agency for International Development", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1231, "sentence": "ltalian red cross worker freed in philippines vagni told abs-cbn network that he often imagined seeing `` my head in a big basket. ''", "head_entity": "he", "tail_entity": "Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1249, "sentence": "adela dworin, deputy speaker among the `jewish patronage , '' said ,` i am very sorry for what happened to gross, '' adding that `` the saddest thing is that he wanted to engage the jewish community, which is totally strange to his activities. ''", "head_entity": "Adela Dworin", "tail_entity": "Jewish Patronage", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1251, "sentence": "toward nov, baldino, whom lived toward northwestern huntington, ap., did harvested a authorization from deficit for her postures because cephalon ' s chairman and chief executive.", "head_entity": "Baldino", "tail_entity": "Cephalon", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1254, "sentence": "but taught chen tao, of with beijing-based china youth university for political sciences, said the public should focus on problems behind the case rather than the reputations of celebrities.", "head_entity": "Chen Tao", "tail_entity": "China Youth University for Political Sciences", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1256, "sentence": "abn amro heralded sonntag that it will nominate royal bank of scotland-lrb-rbs-rrb-executive mark fisher as its new chairman, following the takeover of the dutch lender by the rbs-led consortium.", "head_entity": "Mark Fisher", "tail_entity": "ABN Amro", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1259, "sentence": "`` avarice, '' by former new york times reporter charlie leduff, focuses on his hometown of detroit-lrb-he now works for the detroit news, presumably less villainous a player in the msm-rrb-.", "head_entity": "his", "tail_entity": "Detroit News", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1261, "sentence": "vagni were one of three international committee of the red cross workers kidnapped on jolo, an impoverished muslim region 590 miles-lrb-950 kilometers-rrb-south of manila, on jan 15.", "head_entity": "Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1263, "sentence": "two among jupp ' s managers at the defense, scientific and technology laboratory were charged with manslaughter following an inquiry into the blast.", "head_entity": "Jupp", "tail_entity": "Defense", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1312, "sentence": "the russian competition vladimir ladyzhenskiy nor the finnish compete timo kaukonen became unconscious when the competition has lasted for six minutes.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Finnish", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 1435, "sentence": "one imaginable interpretation makes the talisman from prodigious cents throughout functionaries whom strike prodigious refers, argued nell minow, publisher nor co-founder of the corporate library.", "head_entity": "Corporate Library", "tail_entity": "co", "predicted_relation": "org:founded_by", "predicted_relation_id": 41}

{"index": 1479, "sentence": "relying to faraco and a friend of samudio the footballer had threatened her during her pregnancy and demanded that she have an abortion.", "head_entity": "her", "tail_entity": "footballer", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 1686, "sentence": "it argued the accuse, jon graham, 55, shot her because the activist group ' s leaders thought she was a government informant.", "head_entity": "she", "tail_entity": "activist", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 1782, "sentence": "currently is the proof: nir rosen resigns after remarks about cbs reporter lara logan rosen, a fellow reporter, has been forced to resign his position at nyu over comments he made regarding logan.", "head_entity": "his", "tail_entity": "NYU", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 1961, "sentence": "additionally unless the stockpile market has performed well this year, hodgson of the corporate library says he thinks that many directors will face significant opposition, albeit still toothless, from shareholders at annual elections in 2010.", "head_entity": "Corporate Library", "tail_entity": "Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}

{"index": 2011, "sentence": "adela dworin, deputy presidency of the `jewish patronage , '' said ,` i am very sorry for what happened to gross, '' adding that `` the saddest thing is that he wanted to engage the jewish community, which is totally strange to his activities. ''", "head_entity": "Adela Dworin", "tail_entity": "Patronage", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 2135, "sentence": "convened felony boss xie caiping, labeled the `` godmother of the underworld '' in southwest china ' s chongqing municipality, was sentenced 18 years in prison after a first-instance trial on tuesday.", "head_entity": "Xie Caiping", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 2331, "sentence": "also, she had never have the haitian newspapers obligated to assume the kiddies out of the country, and a dominican diplomat told the ap he warned her the day the missionaries were arrested that without those papers she could be arrested.", "head_entity": "her", "tail_entity": "diplomat", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 2552, "sentence": "for the 2007 zogby ballot, regarding the two primary schools of thought: let it happen on purpose-lrb-lihop-rrb-the us government knew the attacks were coming but consciously let them proceed for various political, military and economic motives made it happen on purpose-lrb-mihop-rrb-certain us government elements actively planned or assisted some aspects of the attacks group lihop mihop progressive 35.8 1.7 liberal 34.4 5.0 moderate 27 5.7 conservative 22.3 4.1 veryconservative 5.9 1.0 libertarian 18.8 11.6 [http://www.911truth.org/images/zogbypoll2007.pdf](http://www.911truth.org/images/zogbypoll2007.pdf)", "head_entity": "LIHOP", "tail_entity": "US", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}

{"index": 2812, "sentence": "ultimately, a sympathetic guard shared his cell phone and she called her father back in texas, who called his congressman, ted poe, for help.", "head_entity": "his", "tail_entity": "Texas", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 2994, "sentence": "j did been working during the asuncion that craig ranke-lrb-lyte trip-rrb-admitted that he had no intention of ever taking his fantastic rubbish to a real news outlet during the month of august, 2007.", "head_entity": "Craig Ranke", "tail_entity": "Lyte Trip", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 3123, "sentence": "9 / 11--ltalian says 9-11 solved--manpower of american free press, fri dec 14 23: 28 italian says 9-11 solved it ' s common knowledge, he reveals, cia, mossad behind terror attacks by the staff of american free press [http://www.americanfreepress.net/html/9-11solved118.html](http://www.americanfreepress.net/html/9-11solved118.html) former italian president francesco cossiga, who revealed the existence of operation gladio, has told italy ' s oldest and most... more it ' s true... all true-israel did it--efriam halevy, sat dec 15 19: 06", "head_entity": "American Free Press", "tail_entity": "Efriam Halevy", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 3410, "sentence": "the sol satellite lk concerto fest in central taiwan ' s nantou county will kick off oct. 8 and present a series of firework and live music shows, including a performance by the national taiwan symphony orchestra, organizers said wednesday.", "head_entity": "National Taiwan Symphony Orchestra", "tail_entity": "Taiwan", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}

{"index": 3477, "sentence": "the assistant enquiries director of el salvador ' s police, howard augusto cotto, told reporters friday that he was seeking fingerprints for puello to determine whether he is the 32-year-old salvadoran named in an interpol arrest warrant for allegedly running a sex trafficking ring that lured women and girls from the caribbean and central america into prostitution with bogus offers of modeling jobs.", "head_entity": "he", "tail_entity": "Salvadoran", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 3809, "sentence": "aig, which did stored for collapse per the us vlado toward juni 2008, heralded previous mois he did vending aia to cautious and another unit, alico, to us rival metlife to pay back a huge chunk of its government bailout.", "head_entity": "ALICO", "tail_entity": "US", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}

{"index": 3825, "sentence": "aime joneses is free to file criminal charges in her home state against the govt, and / or individuals.", "head_entity": "her", "tail_entity": "State", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 3851, "sentence": "neither syndicate owns avowed responsibility for his kidnapping and officials said they have no word on his whereabouts.", "head_entity": "his", "tail_entity": "kidnapping", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 3902, "sentence": "a position j read across the jref archives yesterday said that dylan avery was perplexed when a british interviewer explained to him the difference between saying something is `like ' something else and saying something` is ' the object of comparison.", "head_entity": "him", "tail_entity": "British", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 4011, "sentence": "`` meredith furthermore leaves a prodigious peephole in our lives and her presence is missed every time we meet up as a family, '' john kercher jr., one of her brothers, told a press conference in perugia.", "head_entity": "her", "tail_entity": "John Kercher Jr.", "predicted_relation": "per:siblings", "predicted_relation_id": 35}

{"index": 4159, "sentence": "researching cloud, who was convicted of murder in 2004, testified that he, graham and clark kidnapped aquash from denver and took her to rapid city.", "head_entity": "her", "tail_entity": "murder", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}

{"index": 4679, "sentence": "im neither genius or architect, but if i was juror and his presentation was part of the case for the prosecution, i would be looking forward to the case for the defence.", "head_entity": "his", "tail_entity": "architect", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 4876, "sentence": "the overall thing reminded her of first husband michael lomax, whose run for fulton county commission in the late 1970s caused the opposition to put out handbills saying lomax was the son-in-law of a black radical, cleage says.", "head_entity": "Michael Lomax", "tail_entity": "Lomax", "predicted_relation": "per:spouse", "predicted_relation_id": 37}

{"index": 4912, "sentence": "`` we assuredly see it as a victory, '' said kunstler, the daughter of william kunstler, the colorful crusading civil rights lawyer who died in 1995.", "head_entity": "William Kunstler", "tail_entity": "1995", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}

{"index": 4928, "sentence": "youssoufou bamba, the ouattara goverment ' s first officially recognized envoy, said after presenting his credentials wednesday in new york that he appealed to secretary-general ban ki-moon for action by u. n. peacekeepers to stop what he called `` massive violations of human rights. ''", "head_entity": "Youssoufou Bamba", "tail_entity": "Ouattara", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 5128, "sentence": "scagliotti, already 40, thereafter shared the commodity among child amie mario gonzalez of canoga park.", "head_entity": "Mario Gonzalez", "tail_entity": "40", "predicted_relation": "per:age", "predicted_relation_id": 4}

{"index": 5301, "sentence": "where had i morph back across craig ranke-formerly j was aldo marquis and then `` a close friend of aldo ''.", "head_entity": "Craig Ranke", "tail_entity": "Aldo Marquis", "predicted_relation": "NA", "predicted_relation_id": 13}

{"index": 5339, "sentence": "the suspicion toward the slain of meredith kercher, 21-year-old rudy hermann guede from ivory coast, was detained after he travelled by train between the western german cities of mainz and koblenz.", "head_entity": "Meredith Kercher", "tail_entity": "German", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 5419, "sentence": "the bar advocating american amanda knox from charges of involvement in the 2007 sex murder of her british housemate in italy made an emotional appeal wednesday for her acquittal.", "head_entity": "her", "tail_entity": "American", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 5459, "sentence": "the expenditure from purifying up existing waste is higher than previously thought the uk ' s nuclear waste clean-up programme could cost more than # 70bn, according to the nuclear decommissioning authority-lrb-nda-rrb-.", "head_entity": "NDA", "tail_entity": "UK", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}

{"index": 5518, "sentence": "toward 2002, matias kings, servicing a penitentiary period throughout manslaughter, explained he had committed the rape and acted alone.", "head_entity": "he", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 5656, "sentence": "even, because irrespective journalism nir rosen described vividly and agonizingly in his indispensable book, in the belly of the green bird, they involve a kind of energetic brutality that is only occasionally reported by an embedded american mainstream journalist.", "head_entity": "his", "tail_entity": "journalist", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 5962, "sentence": "procter & gamble purchases men ' s skin care brand zirh", "head_entity": "Zirh", "tail_entity": "Procter & Gamble", "predicted_relation": "org:parents", "predicted_relation_id": 15}

{"index": 5971, "sentence": "reviewer ' s pio: wayne a. holst teaches religion and culture at the university of calgary and helps facilitate adult spiritual development at st. david ' s united church, calgary.", "head_entity": "Wayne A. Holst", "tail_entity": "Reviewer", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 6164, "sentence": "`` countrywide presents a few prospect throughout ribera from americano to add what we believe makes the highest internal borrower tribune through an attractive fee nor to say our post because the state ' s primo borrowing to abusers, '' ribera among america chief executive ken lewis said in a statement.", "head_entity": "Countrywide", "tail_entity": "Ken Lewis", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}

{"index": 6187, "sentence": "relying to a february 1993 recent ny hours paragraphs, impeached hamas terrorist device nor previous uasr employee, mohammed salah, told federal authorities that uasr served as `` the political command of hamas in the united states ''.", "head_entity": "UASR", "tail_entity": "Mohammed Salah", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}

{"index": 6194, "sentence": "for the 2007 zogby vote, respecting the two crucial campuses from brainchild: let it happen on purpose-lrb-lihop-rrb-the us government knew the attacks were coming but consciously let them proceed for various political, military and economic motives made it happen on purpose-lrb-mihop-rrb-certain us government elements actively planned or assisted some aspects of the attacks group lihop mihop progressive 35.8 1.7 liberal 34.4 5.0 moderate 27 5.7 conservative 22.3 4.1 veryconservative 5.9 1.0 libertarian 18.8 11.6 [http://www.911truth.org/images/zogbypoll2007.pdf](http://www.911truth.org/images/zogbypoll2007.pdf)", "head_entity": "LIHOP", "tail_entity": "US", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}

{"index": 6403, "sentence": "his murder was onto six mes than two fbi retainers became beheaded down toward a battaglia among aim congressmen, nor two decades than she joined toward meta ' s 71-date profession of the south dakota reservation town of wounded knee.", "head_entity": "she", "tail_entity": "South Dakota", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}

{"index": 6586, "sentence": "a amigo introduced her to his brother, dan lafreniere, and hagemo found herself in her first serious relationship.", "head_entity": "her", "tail_entity": "Dan LaFreniere", "predicted_relation": "per:spouse", "predicted_relation_id": 37}

{"index": 6595, "sentence": "`` nothin from those chiefs have totalled up nor argued, ` we gave a miscalculation here by not calling management to account, '' ' said paul hodgson, senior research associate at the corporate library, a corporate governance research firm.", "head_entity": "Corporate Library", "tail_entity": "Paul Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}

{"index": 6797, "sentence": "`` where you ' re debating onto reinstating credibility and a sense of fairness, i think there is a tremendous symbolic effect that ' s legitimate and worthwhile, '' said nell minow, editor at research firm corporate library.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}

{"index": 6903, "sentence": "convened felony boss xie caiping, labeled the `` godmother of the underworld '' in southwest china ' s chongqing municipality, was sentenced 18 years in prison tuesday.", "head_entity": "Xie Caiping", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 7063, "sentence": "a urban justice toward nanjing convicted xie caiping--sister-in-law of the former head of the city ' s judiciary, wen qiang--of a series of crimes linked to the gambling syndicate, said the verdict, posted on the court ' s website.", "head_entity": "Xie Caiping", "tail_entity": "Wen Qiang", "predicted_relation": "per:other_family", "predicted_relation_id": 38}

{"index": 7538, "sentence": "vladimir ladyzhenskiy among russia died after she suffered a shock in the final of the spa world championship in heinola, a southern city of finland, on saturday.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Finland", "predicted_relation": "per:country_of_death", "predicted_relation_id": 18}

{"index": 7554, "sentence": "boss joseph cerniglia, a contestant on the reality cooking show `` kitchen nightmares, '' also jumped from the iconic bridge in the past two weeks.", "head_entity": "Joseph Cerniglia", "tail_entity": "Kitchen Nightmares", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 7993, "sentence": "`playback time is over , '' declared charles ble goude , gbagbo ' s minister for youth , who has been under un sanctions since 2006 for` acts of violence by street militias, including beatings, rapes and extrajudicial killings ''.", "head_entity": "Charles Ble Goude", "tail_entity": "violence", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 8064, "sentence": "lomax recalls where, at age 13, his mother moved him and his five brothers and sisters from los angeles, california, to tuskegee, alabama, to cover the civil rights movement for nation magazine.", "head_entity": "Lomax", "tail_entity": "brothers", "predicted_relation": "per:siblings", "predicted_relation_id": 35}

{"index": 8151, "sentence": "further an conflicting congregate from christian nor rabbinical clerical will petition the irs thu to stop the outcry before it starts, calling the adf ' s `` pulpit initiative '' an assault on the rule of law and the separation of church and state.", "head_entity": "ADF", "tail_entity": "Christian", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}

{"index": 8321, "sentence": "two from jupp ' s managers at the defense, scientific and technology laboratory were charged with manslaughter following an inquiry into the blast.", "head_entity": "Jupp", "tail_entity": "manslaughter", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 8383, "sentence": "an current miscommunication makes that heloise nor abelard was nun and monk when they became lovers; there ' s a certain appeal there for some people.", "head_entity": "Heloise", "tail_entity": "nun", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 8462, "sentence": "the writ nor definitive apprehend comes one day after bruno ' s 17-year-old cousin told police that he participated in the kidnapping of samudio and claimed that she is dead.", "head_entity": "she", "tail_entity": "17", "predicted_relation": "per:age", "predicted_relation_id": 4}

{"index": 8468, "sentence": "please signal during to the complaint: `` convention would investigate the rape of jamie leigh jones and others, hold those involved accountable, and bring us contractors under the jurisdiction of us law. ''", "head_entity": "Jamie Leigh Jones", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}

{"index": 8881, "sentence": "eugene, an counsel, did sequestered for her bureau october. 21--smaller upon three jours than her hermano patricia gonzalez stepped down as chihuahua state ' s attorney general due to a change of governor.", "head_entity": "Patricia Gonzalez", "tail_entity": "governor", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 9222, "sentence": "it also plans to continue civic work with the greater houston partnership, which he served as chairman, and the national urban league.", "head_entity": "National Urban League", "tail_entity": "he", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}

{"index": 9336, "sentence": "american remy knox during mardi indicated she did harmless from the 2007 gender-manslaughter among her british housemate in an italian university town as she took the stand for the first time in the trial.", "head_entity": "she", "tail_entity": "Italian", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 9471, "sentence": "ltalian juries withdrew friday to decide whether 22-year-old american amanda knox should spend the rest of her days in prison for the 2007 murder of her british housemate.", "head_entity": "her", "tail_entity": "British", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 9491, "sentence": "the stronger fund that rolled across piedra ' s coral gables practise, the more he gave to scientology groups.", "head_entity": "Piedra", "tail_entity": "Coral Gables", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}

{"index": 9666, "sentence": "everything presently heeded travis the orangutan, whose landowners propelled it around in a tow truck, and police officer frank chiafari, on the job for 25 years, remembered playing with him when their paths would cross.", "head_entity": "him", "tail_entity": "Police Officer", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 9704, "sentence": "one among the crew, chen tao, from chinese mainland ' s liaoning province, was killed by the pirates during his captivity in june, the owner of the ship confirmed to xinhua on wednesday.", "head_entity": "Chen Tao", "tail_entity": "Liaoning Province", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}

{"index": 9780, "sentence": "per the retrial the seattle native accused police of beating and intimidating her during questioning after the gruesome murder.", "head_entity": "her", "tail_entity": "Seattle", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}

{"index": 9926, "sentence": "ishmael rukwago, a upper adf comandante based in the democratic republic of congo, denied any involvement.", "head_entity": "ADF", "tail_entity": "Democratic Republic of Congo", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}

{"index": 10065, "sentence": "the different did a old challenger on his `` hell ' s kitchen '' chef competition.", "head_entity": "his", "tail_entity": "chef", "predicted_relation": "per:title", "predicted_relation_id": 30}

{"index": 10407, "sentence": "`` it ' s implementing throughout chairwoman of the united states, not president of the urban league, '' said jabari asim, editor of the crisis, the naacp magazine, echoing the comments of several prominent blacks.", "head_entity": "Urban League", "tail_entity": "United States", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}

{"index": 10684, "sentence": "dutch newsprint babants dagblad said the boy was likely nine-year-old ruben van assouw from tilburg in the southern netherlands who had been on safari in south africa with his mother trudy, 41, father patrick, 40, and his brother enzo, 11.", "head_entity": "Ruben van Assouw", "tail_entity": "40", "predicted_relation": "per:age", "predicted_relation_id": 4}

{"index": 10718, "sentence": "punctually eagerly to 2003 where jean graham, a nationality from the cava nor aishihik preliminary nationals of the yukon, canada and a canadian citizen, along with co-accused arlo looking cloud, is charged with the 28-year-old murder of nova scotia mi ` kmaq anna mae pictou.", "head_entity": "Anna Mae Pictou", "tail_entity": "Canadian", "predicted_relation": "per:origin", "predicted_relation_id": 27}

{"index": 10847, "sentence": "independence journalist nir rosen, who has been living in the real iraq unembedded, lets loose with what the sunni heartland of iraq is actually like under us occupation.", "head_entity": "Nir Rosen", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}

{"index": 11018, "sentence": "`` j learn for cadets nor i challenge them, '' says heloise, 58, whom became over the dynasty councils undertaking where his mummy, also named heloise, died in 1977.", "head_entity": "Heloise", "tail_entity": "1977", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}

{"index": 11138, "sentence": "ltalian credentials are concluding onto on sicilian mafia supremo matteo messina denaro, interior minister roberto maroni said monday as he announced the arrest of denaro ' s brother and 18 associates.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Interior", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}

{"index": 11222, "sentence": "the accusers alleged that between may 2004 and 2009, wang also offered bribes well over 1. 28 million yuan to five government employees, including wen qiang, formerly chief of chongqing municipal bureau of justice, and li hanbin, a former police chief of the chongqing municipal bureau of public security who was assigned to crack down on organized crimes.", "head_entity": "Wen Qiang", "tail_entity": "police chief", "predicted_relation": "per:title", "predicted_relation_id": 30}

所有样本已分析完毕。