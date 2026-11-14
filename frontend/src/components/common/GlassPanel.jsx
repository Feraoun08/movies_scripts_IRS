export default function GlassPanel({ as: Tag = 'div', className = '', children, ...rest }) {
  return (
    <Tag className={`glass ${className}`.trim()} {...rest}>
      {children}
    </Tag>
  );
}
