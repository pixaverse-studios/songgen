from songgen.tokenizers.lyrics.lyrics_tokenizer import VoiceBpeTokenizer

def test_lyrics_cleaning():
    tokenizer = VoiceBpeTokenizer()
    
    test_cases = [
        # Test case 1: Timestamps and music notations
        (
            "[0:22] ♪♪♪\n[0:29] ♪ YOU SAY YOU LOVE ME\nBUT I DON'T CARE ♪",
            "YOU SAY YOU LOVE ME BUT I DON'T CARE"
        ),
        
        # Test case 2: Multiple types of brackets
        (
            "[Verse 1] (Chorus) [Music] Hello World",
            "Hello World"
        ),
        
        # Test case 3: Multiple newlines and spaces
        (
            "Line 1\n\n\nLine 2    Line 3",
            "Line 1\nLine 2 Line 3"
        ),
        
        # Test case 4: Complex example
        (
            "[0:22] ♪♪♪\n[0:29] ♪ YOU SAY YOU LOVE ME\nBUT I DON'T CARE ♪\n[0:34] ♪ THAT I BROKE MY HAND ON THE\nSAME WALL THAT YOU TOLD ME\nTHAT HE FUCKED YOU ON ♪\n[0:40] ♪ YOU THINK IT'S SO EASY ♪\n[0:43] ♪ FUCKIN' WITH MY FEELINGS ♪",
            "YOU SAY YOU LOVE ME BUT I DON'T CARE THAT I BROKE MY HAND ON THE SAME WALL THAT YOU TOLD ME THAT HE FUCKED YOU ON YOU THINK IT'S SO EASY FUCKIN' WITH MY FEELINGS"
        )
    ]
    
    for input_lyrics, expected_output in test_cases:
        cleaned_lyrics = tokenizer.clean_lyrics(input_lyrics)
        assert cleaned_lyrics == expected_output, f"\nExpected: {expected_output}\nGot: {cleaned_lyrics}"
        
def print_example():
    """Interactive example to see the cleaning in action"""
    tokenizer = VoiceBpeTokenizer()
    
    # Test case with timestamps, music notations, and other special characters
    input_lyrics = """[0:22] ♪♪♪
[0:29] ♪ YOU SAY YOU LOVE ME
BUT I DON'T CARE ♪
[0:34] ♪ THAT I BROKE MY HAND ON THE
SAME WALL THAT YOU TOLD ME
THAT HE FUCKED YOU ON ♪
[0:40] ♪ YOU THINK IT'S SO EASY ♪
[0:43] ♪ FUCKIN' WITH MY FEELINGS ♪"""

    cleaned_lyrics = tokenizer.clean_lyrics(input_lyrics)
    print("Original lyrics:")
    print(input_lyrics)
    print("\nCleaned lyrics:")
    print(cleaned_lyrics)

if __name__ == "__main__":
    # Run the test
    test_lyrics_cleaning()
    print("\nAll tests passed!")
    
    # Show an example
    print("\nExample cleaning:")
    print_example() 