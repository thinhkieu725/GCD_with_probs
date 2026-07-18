-- Token-ID-style concrete of Seq with letter-prefixed terminals ("t100"),
-- the encoding required by the probability runtime: digit-only terminals
-- are lexed as Int literals by the GF C runtime, which breaks complete().
concrete SeqTok of Seq = {
  lincat S = Str ;
  lin
    Short = "t100" ++ "t200" ;
    Long = "t100" ++ "t300" ++ "t400" ;
}
