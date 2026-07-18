-- Toy grammar with numeric terminals, standing in for LM token IDs.
-- Generates either "100 200" or "100 300 400".
abstract Seq = {
  flags startcat = S ;
  cat S ;
  fun
    Short : S ;
    Long : S ;
}
