# Update the report's contents / figure / table lists in Word, save, and export a PDF.
#   powershell -ExecutionPolicy Bypass -File scripts\tarp_finalize.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\tarp_finalize.ps1 -Docx some\other\copy.docx
#
# The list entries get single line spacing so the contents, figure and table lists each
# fit their page. If Word is already open with your own documents, COM may hand back that
# same Word; then this script leaves the window alone and does not quit it.
param([string]$Docx = "docs\tarp_report\TARP_Report_StanceSense-RT.docx")
$path = (Resolve-Path $Docx).Path
$pdf = [System.IO.Path]::ChangeExtension($path, ".pdf")
$word = New-Object -ComObject Word.Application
$ours = ($word.Documents.Count -eq 0)          # a fresh, private Word instance
if ($ours) { $word.Visible = $false }
try {
    $doc = $word.Documents.Open($path, $false, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    foreach ($id in -20, -21) {                  # wdStyleTOC1, wdStyleTOC2
        $pf = $doc.Styles.Item($id).ParagraphFormat
        $pf.LineSpacingRule = 0                  # single
        $pf.SpaceBefore = 0
        $pf.SpaceAfter = 2
    }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }   # page numbers settle after the first pass
    $doc.Save()
    $doc.SaveAs([ref]$pdf, [ref]17)
    $doc.Close($false)
    "updated + PDF: $pdf"
}
finally {
    if ($ours) { $word.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    [GC]::Collect()
}
