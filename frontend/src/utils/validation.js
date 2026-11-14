export function validateFilters({ year, minRating, maxRating }) {
  if (year !== '') {
    const value = Number(year);
    if (!Number.isInteger(value) || value < 1880 || value > 2100) {
      return 'Year must be a valid whole number.';
    }
  }

  for (const rating of [minRating, maxRating]) {
    if (rating !== '' && (Number(rating) < 0 || Number(rating) > 10)) {
      return 'Ratings must be between 0 and 10.';
    }
  }

  if (minRating !== '' && maxRating !== '' && Number(minRating) > Number(maxRating)) {
    return 'Minimum rating cannot be greater than maximum rating.';
  }

  return '';
}
