import { readStations } from 'db-stations'
import fs from 'node:fs'

async function main () {
  const outputPath = 'data/stations_dataset.csv'
  const out = fs.createWriteStream(outputPath, { encoding: 'utf8' })

  // CSV header
  out.write('type,id,nr,name,city\n')

  for await (const station of readStations()) {
    const type = station.type ?? ''
    const id = station.id ?? ''
    const nr = station.nr ?? ''
    const name = station.name ?? ''
    const city = station.address?.city ?? ''

    // Basic CSV-safe escaping: wrap in quotes and escape inner quotes
    const cells = [type, id, nr, name, city].map((value) => {
      const str = String(value).replace(/"/g, '""')
      return `"${str}"`
    })

    out.write(cells.join(',') + '\n')
  }

  out.end()
  console.log(`Wrote dataset to ${outputPath}`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})


